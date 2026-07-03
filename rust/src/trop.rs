//! TROP (Triply Robust Panel) estimator acceleration.
//!
//! This module provides optimized implementations of:
//! - Pairwise unit distance matrix computation (parallelized)
//! - LOOCV grid search (parallelized across parameter combinations)
//! - Bootstrap variance estimation (parallelized across iterations)
//!
//! Reference:
//! Athey, S., Imbens, G. W., Qu, Z., & Viviano, D. (2025). Triply Robust
//! Panel Estimators. Working Paper.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::linalg::ndarray_to_faer;

/// Minimum chunk size for parallel distance computation.
/// Reduces scheduling overhead for small matrices.
const MIN_CHUNK_SIZE: usize = 16;

/// Compute pairwise unit distance matrix using parallel RMSE computation.
///
/// Following TROP Equation 3 (page 7):
/// dist_unit(j, i) = sqrt(Σ_u (Y_{iu} - Y_{ju})² / n_valid)
///
/// Only considers valid observations where both units have D=0 (control)
/// and non-NaN values.
///
/// # Arguments
/// * `y` - Outcome matrix (n_periods x n_units)
/// * `d` - Treatment indicator matrix (n_periods x n_units), 0=control, 1=treated
///
/// # Returns
/// Distance matrix (n_units x n_units) where [j, i] = RMSE distance from j to i.
/// Diagonal is 0, pairs with no valid observations get inf.
#[pyfunction]
pub fn compute_unit_distance_matrix<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    d: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y_arr = y.as_array();
    let d_arr = d.as_array();

    let dist_matrix = compute_unit_distance_matrix_internal(&y_arr, &d_arr);

    Ok(dist_matrix.to_pyarray(py))
}

/// Internal implementation of unit distance matrix computation.
///
/// Parallelizes over unit pairs using rayon.
fn compute_unit_distance_matrix_internal(y: &ArrayView2<f64>, d: &ArrayView2<f64>) -> Array2<f64> {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    // Create validity mask: (D == 0) & !isnan(Y)
    // Shape: (n_periods, n_units)
    let valid_mask: Array2<bool> = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
        d[[t, i]] == 0.0 && y[[t, i]].is_finite()
    });

    // Pre-compute Y values with invalid entries set to NaN
    let y_masked: Array2<f64> = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
        if valid_mask[[t, i]] {
            y[[t, i]]
        } else {
            f64::NAN
        }
    });

    // Transpose to (n_units, n_periods) for row-major access
    let y_t = y_masked.t();
    let valid_t = valid_mask.t();

    // Initialize output matrix
    let mut dist_matrix = Array2::<f64>::from_elem((n_units, n_units), f64::INFINITY);

    // Set diagonal to 0
    for i in 0..n_units {
        dist_matrix[[i, i]] = 0.0;
    }

    // Compute upper triangle in parallel, then mirror
    // We parallelize over rows (unit j) and compute all pairs (j, i) for i > j
    let row_results: Vec<Vec<(usize, f64)>> = (0..n_units)
        .into_par_iter()
        .with_min_len(MIN_CHUNK_SIZE)
        .map(|j| {
            let mut pairs = Vec::with_capacity(n_units - j - 1);

            for i in (j + 1)..n_units {
                let dist = compute_pair_distance(
                    &y_t.row(j),
                    &y_t.row(i),
                    &valid_t.row(j),
                    &valid_t.row(i),
                );
                pairs.push((i, dist));
            }

            pairs
        })
        .collect();

    // Fill matrix from parallel results
    for (j, pairs) in row_results.into_iter().enumerate() {
        for (i, dist) in pairs {
            dist_matrix[[j, i]] = dist;
            dist_matrix[[i, j]] = dist; // Symmetric
        }
    }

    dist_matrix
}

/// Compute RMSE distance between two units over valid periods.
///
/// Returns infinity if no valid overlapping observations exist.
#[inline]
fn compute_pair_distance(
    y_j: &ArrayView1<f64>,
    y_i: &ArrayView1<f64>,
    valid_j: &ArrayView1<bool>,
    valid_i: &ArrayView1<bool>,
) -> f64 {
    let n_periods = y_j.len();
    let mut sum_sq = 0.0;
    let mut n_valid = 0usize;

    for t in 0..n_periods {
        if valid_j[t] && valid_i[t] {
            let diff = y_i[t] - y_j[t];
            sum_sq += diff * diff;
            n_valid += 1;
        }
    }

    if n_valid > 0 {
        (sum_sq / n_valid as f64).sqrt()
    } else {
        f64::INFINITY
    }
}

/// Perform univariate LOOCV search over a single parameter.
///
/// Following paper's footnote 2, this performs a grid search for one parameter
/// while holding others fixed. Used in the two-stage LOOCV approach.
///
/// # Arguments
/// * `y` - Outcome matrix (n_periods x n_units)
/// * `d` - Treatment indicator matrix
/// * `control_mask` - Boolean mask for control observations
/// * `time_dist` - Time distance matrix
/// * `control_obs` - List of control observations for LOOCV
/// * `grid` - Grid of values to search
/// * `fixed_time` - Fixed lambda_time (0.0 for uniform weights)
/// * `fixed_unit` - Fixed lambda_unit (0.0 for uniform weights)
/// * `fixed_nn` - Fixed lambda_nn (inf to disable factor model)
/// * `param_type` - Which parameter to search: 0=time, 1=unit, 2=nn
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
/// (best_value, best_score)
fn univariate_loocv_search(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    control_obs: &[(usize, usize)],
    grid: &[f64],
    fixed_time: f64,
    fixed_unit: f64,
    fixed_nn: f64,
    param_type: usize, // 0=time, 1=unit, 2=nn
    max_iter: usize,
    tol: f64,
) -> (f64, f64) {
    let mut best_score = f64::INFINITY;
    let mut best_value = grid.first().copied().unwrap_or(0.0);

    // Parallelize over grid values
    let results: Vec<(f64, f64)> = grid
        .par_iter()
        .map(|&value| {
            // Convert λ_nn=∞ → 1e10 (factor model disabled, L≈0).
            // λ_time and λ_unit use 0.0 for uniform weights per Eq. 3 (no inf conversion).
            let (lambda_time, lambda_unit, lambda_nn) = match param_type {
                0 => {
                    // Searching λ_time: use grid value directly (no inf expected)
                    (
                        value,
                        fixed_unit,
                        if fixed_nn.is_infinite() {
                            1e10
                        } else {
                            fixed_nn
                        },
                    )
                }
                1 => {
                    // Searching λ_unit: use grid value directly (no inf expected)
                    (
                        fixed_time,
                        value,
                        if fixed_nn.is_infinite() {
                            1e10
                        } else {
                            fixed_nn
                        },
                    )
                }
                _ => {
                    // Searching λ_nn: convert inf → 1e10 (factor model disabled)
                    let value_converted = if value.is_infinite() { 1e10 } else { value };
                    (fixed_time, fixed_unit, value_converted)
                }
            };

            let (score, _, _) = loocv_score_for_params(
                y,
                d,
                control_mask,
                time_dist,
                control_obs,
                lambda_time,
                lambda_unit,
                lambda_nn,
                max_iter,
                tol,
            );
            (value, score)
        })
        .collect();

    for (value, score) in results {
        if score < best_score {
            best_score = score;
            best_value = value;
        }
    }

    (best_value, best_score)
}

/// Cycle through parameters until convergence (coordinate descent).
///
/// Following paper's footnote 2 (Stage 2), iteratively optimize each parameter.
fn cycling_parameter_search(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    control_obs: &[(usize, usize)],
    lambda_time_grid: &[f64],
    lambda_unit_grid: &[f64],
    lambda_nn_grid: &[f64],
    initial_time: f64,
    initial_unit: f64,
    initial_nn: f64,
    max_iter: usize,
    tol: f64,
    max_cycles: usize,
) -> (f64, f64, f64) {
    let mut lambda_time = initial_time;
    let mut lambda_unit = initial_unit;
    let mut lambda_nn = initial_nn;
    let mut prev_score = f64::INFINITY;

    for _cycle in 0..max_cycles {
        // Optimize λ_unit (fix λ_time, λ_nn)
        let (new_unit, _) = univariate_loocv_search(
            y,
            d,
            control_mask,
            time_dist,
            control_obs,
            lambda_unit_grid,
            lambda_time,
            0.0,
            lambda_nn,
            1,
            max_iter,
            tol,
        );
        lambda_unit = new_unit;

        // Optimize λ_time (fix λ_unit, λ_nn)
        let (new_time, _) = univariate_loocv_search(
            y,
            d,
            control_mask,
            time_dist,
            control_obs,
            lambda_time_grid,
            0.0,
            lambda_unit,
            lambda_nn,
            0,
            max_iter,
            tol,
        );
        lambda_time = new_time;

        // Optimize λ_nn (fix λ_unit, λ_time)
        let (new_nn, score) = univariate_loocv_search(
            y,
            d,
            control_mask,
            time_dist,
            control_obs,
            lambda_nn_grid,
            lambda_time,
            lambda_unit,
            0.0,
            2,
            max_iter,
            tol,
        );
        lambda_nn = new_nn;

        // Check convergence
        if (score - prev_score).abs() < 1e-6 {
            break;
        }
        prev_score = score;
    }

    (lambda_time, lambda_unit, lambda_nn)
}

/// Perform LOOCV grid search over tuning parameters using two-stage approach.
///
/// Following paper's footnote 2:
/// - Stage 1: Univariate searches for initial values with extreme fixed parameters
/// - Stage 2: Cycling (coordinate descent) until convergence
///
/// Following TROP Equation 5 (page 8):
/// Q(λ) = Σ_{j,s: D_js=0} [τ̂_js^loocv(λ)]²
///
/// # Arguments
/// * `y` - Outcome matrix (n_periods x n_units)
/// * `d` - Treatment indicator matrix (n_periods x n_units)
/// * `control_mask` - Boolean mask (n_periods x n_units) for control observations
/// * `time_dist_matrix` - Pre-computed time distance matrix (n_periods x n_periods)
/// * `lambda_time_grid` - Grid of time decay parameters
/// * `lambda_unit_grid` - Grid of unit distance parameters
/// * `lambda_nn_grid` - Grid of nuclear norm parameters
/// * `max_iter` - Maximum iterations for model estimation
/// * `tol` - Convergence tolerance
///
/// # Returns
/// (best_lambda_time, best_lambda_unit, best_lambda_nn, best_score, n_valid, n_attempted, first_failed_obs)
/// where n_valid and n_attempted are the counts for the best parameter combination,
/// allowing Python to emit warnings when >10% of fits fail.
/// first_failed_obs is Some((t, i)) if a fit failed during final score computation, None otherwise.
#[pyfunction]
#[pyo3(signature = (y, d, control_mask, time_dist_matrix, lambda_time_grid, lambda_unit_grid, lambda_nn_grid, max_iter, tol))]
#[allow(clippy::too_many_arguments)]
pub fn loocv_grid_search<'py>(
    _py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    d: PyReadonlyArray2<'py, f64>,
    control_mask: PyReadonlyArray2<'py, u8>,
    time_dist_matrix: PyReadonlyArray2<'py, i64>,
    lambda_time_grid: PyReadonlyArray1<'py, f64>,
    lambda_unit_grid: PyReadonlyArray1<'py, f64>,
    lambda_nn_grid: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<(f64, f64, f64, f64, usize, usize, Option<(usize, usize)>)> {
    let y_arr = y.as_array();
    let d_arr = d.as_array();
    let control_mask_arr = control_mask.as_array();
    let time_dist_arr = time_dist_matrix.as_array();
    let lambda_time_vec: Vec<f64> = lambda_time_grid.as_array().to_vec();
    let lambda_unit_vec: Vec<f64> = lambda_unit_grid.as_array().to_vec();
    let lambda_nn_vec: Vec<f64> = lambda_nn_grid.as_array().to_vec();

    // Validate: lambda_time_grid and lambda_unit_grid must not contain inf.
    // Per Athey et al. (2025) Eq. 3: use 0.0 for uniform weights, not inf.
    for &v in &lambda_time_vec {
        if v.is_infinite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "lambda_time_grid must not contain inf. Use 0.0 for uniform weights (disabled) per Athey et al. (2025) Eq. 3."
            ));
        }
    }
    for &v in &lambda_unit_vec {
        if v.is_infinite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "lambda_unit_grid must not contain inf. Use 0.0 for uniform weights (disabled) per Athey et al. (2025) Eq. 3."
            ));
        }
    }

    // Get control observations for LOOCV
    let control_obs = get_control_observations(&y_arr, &control_mask_arr);

    let n_attempted = control_obs.len();

    // Stage 1: Univariate searches for initial values (paper footnote 2)
    // λ_time search: fix λ_unit=0, λ_nn=∞ (disabled)
    let (lambda_time_init, _) = univariate_loocv_search(
        &y_arr,
        &d_arr,
        &control_mask_arr,
        &time_dist_arr,
        &control_obs,
        &lambda_time_vec,
        0.0,
        0.0,
        f64::INFINITY,
        0,
        max_iter,
        tol,
    );

    // λ_nn search: fix λ_time=0 (uniform time weights), λ_unit=0
    let (lambda_nn_init, _) = univariate_loocv_search(
        &y_arr,
        &d_arr,
        &control_mask_arr,
        &time_dist_arr,
        &control_obs,
        &lambda_nn_vec,
        0.0,
        0.0,
        0.0,
        2,
        max_iter,
        tol,
    );

    // λ_unit search: fix λ_nn=∞, λ_time=0
    let (lambda_unit_init, _) = univariate_loocv_search(
        &y_arr,
        &d_arr,
        &control_mask_arr,
        &time_dist_arr,
        &control_obs,
        &lambda_unit_vec,
        0.0,
        0.0,
        f64::INFINITY,
        1,
        max_iter,
        tol,
    );

    // Stage 2: Cycling refinement
    let (best_time, best_unit, best_nn) = cycling_parameter_search(
        &y_arr,
        &d_arr,
        &control_mask_arr,
        &time_dist_arr,
        &control_obs,
        &lambda_time_vec,
        &lambda_unit_vec,
        &lambda_nn_vec,
        lambda_time_init,
        lambda_unit_init,
        lambda_nn_init,
        max_iter,
        tol,
        10,
    );

    // Convert λ_nn=∞ → 1e10 for final score computation (factor model disabled)
    let best_time_eff = best_time;
    let best_unit_eff = best_unit;
    let best_nn_eff = if best_nn.is_infinite() { 1e10 } else { best_nn };

    // Compute final score with converted values
    let (best_score, n_valid, first_failed) = loocv_score_for_params(
        &y_arr,
        &d_arr,
        &control_mask_arr,
        &time_dist_arr,
        &control_obs,
        best_time_eff,
        best_unit_eff,
        best_nn_eff,
        max_iter,
        tol,
    );

    // Return ORIGINAL grid values (for user visibility) but score computed with converted
    Ok((
        best_time,
        best_unit,
        best_nn,
        best_score,
        n_valid,
        n_attempted,
        first_failed,
    ))
}

/// Get all valid control observations for LOOCV.
fn get_control_observations(
    y: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
) -> Vec<(usize, usize)> {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    // Collect all valid control observations
    let mut obs: Vec<(usize, usize)> = Vec::new();
    for t in 0..n_periods {
        for i in 0..n_units {
            if control_mask[[t, i]] != 0 && y[[t, i]].is_finite() {
                obs.push((t, i));
            }
        }
    }

    obs
}

/// Compute LOOCV score for a specific parameter combination.
///
/// # Returns
/// (score, n_valid, first_failed_obs) - the LOOCV score, number of successful fits,
/// and the first failed observation (t, i) if any fit failed, None otherwise.
#[allow(clippy::too_many_arguments)]
fn loocv_score_for_params(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    control_obs: &[(usize, usize)],
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    max_iter: usize,
    tol: f64,
) -> (f64, usize, Option<(usize, usize)>) {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    // Parallelize over control observations — each per-observation computation
    // is independent (compute weight matrix, fit model, extract τ²).
    // with_min_len(64) prevents scheduling overhead from dominating on small panels.
    let (tau_sq_sum, n_valid, first_failed) = control_obs
        .par_iter()
        .with_min_len(64)
        .fold(
            || (0.0f64, 0usize, None::<(usize, usize)>),
            |(sum, valid, first_fail), &(t, i)| {
                let weight_matrix = compute_weight_matrix(
                    y,
                    d,
                    n_periods,
                    n_units,
                    i,
                    t,
                    lambda_time,
                    lambda_unit,
                    time_dist,
                );

                match estimate_model(
                    y,
                    control_mask,
                    &weight_matrix.view(),
                    lambda_nn,
                    n_periods,
                    n_units,
                    max_iter,
                    tol,
                    Some((t, i)),
                ) {
                    Some((alpha, beta, l)) => {
                        let tau = y[[t, i]] - alpha[i] - beta[t] - l[[t, i]];
                        (sum + tau * tau, valid + 1, first_fail)
                    }
                    None => (sum, valid, first_fail.or(Some((t, i)))),
                }
            },
        )
        .reduce(
            || (0.0, 0, None),
            |(s1, v1, f1), (s2, v2, f2)| (s1 + s2, v1 + v2, f1.or(f2)),
        );

    // Per Equation 5: if ANY fit fails, this λ combination is invalid
    if first_failed.is_some() {
        return (f64::INFINITY, n_valid, first_failed);
    }

    if n_valid == 0 {
        (f64::INFINITY, 0, None)
    } else {
        // Return SUM of squared pseudo-treatment effects per Equation 5 (page 8):
        // Q(λ) = Σ_{j,s: D_js=0} [τ̂_js^loocv(λ)]²
        (tau_sq_sum, n_valid, None)
    }
}

/// Compute observation-specific distance from unit j to unit i, excluding target period.
///
/// Issue B fix: Follows Equation 3 (page 7) which specifies 1{u ≠ t} to exclude target period.
fn compute_unit_distance_for_obs(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    j: usize,
    i: usize,
    target_period: usize,
) -> f64 {
    let n_periods = y.nrows();
    let mut sum_sq = 0.0;
    let mut n_valid = 0usize;

    for t in 0..n_periods {
        // Exclude target period (Issue B fix)
        if t == target_period {
            continue;
        }
        // Both units must be control at this period and have valid values
        if d[[t, i]] == 0.0 && d[[t, j]] == 0.0 && y[[t, i]].is_finite() && y[[t, j]].is_finite() {
            let diff = y[[t, i]] - y[[t, j]];
            sum_sq += diff * diff;
            n_valid += 1;
        }
    }

    if n_valid > 0 {
        (sum_sq / n_valid as f64).sqrt()
    } else {
        f64::INFINITY
    }
}

/// Compute observation-specific weight matrix for TROP.
///
/// Time weights: θ_s = exp(-λ_time × |t - s|)
/// Unit weights: ω_j = exp(-λ_unit × dist(j, i))
///
/// Paper alignment notes:
/// - ALL units with `j != target_unit` get distance-based weights
///   (same-cohort donors contribute via their pre-treatment rows)
/// - The (1 - D_js) masking in the loss naturally excludes treated cells
///   via the control mask applied inside `estimate_model`
/// - Weights are unnormalized raw exponentials per REGISTRY Eq. 2/3
/// - Distance excludes target period t per Equation 3
#[allow(clippy::too_many_arguments)]
fn compute_weight_matrix(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    n_periods: usize,
    n_units: usize,
    target_unit: usize,
    target_period: usize,
    lambda_time: f64,
    lambda_unit: f64,
    time_dist: &ArrayView2<i64>,
) -> Array2<f64> {
    // Time weights for this target period: θ_s = exp(-λ_time × |t - s|)
    // Unnormalized per REGISTRY Eq. 2/3.
    let time_weights: Array1<f64> = Array1::from_shape_fn(n_periods, |s| {
        let dist = time_dist[[target_period, s]] as f64;
        (-lambda_time * dist).exp()
    });

    // Unit weights: ω_j = exp(-λ_unit × dist(j, i))
    // Paper alignment: compute for ALL units, let control masking handle exclusion.
    // Unnormalized per REGISTRY Eq. 2/3.
    let mut unit_weights = Array1::<f64>::zeros(n_units);

    if lambda_unit == 0.0 {
        // Uniform weights when lambda_unit = 0
        // All units get weight 1 (control masking will handle exclusion)
        unit_weights.fill(1.0);
    } else {
        // Compute per-observation distance for all units (excluding target unit itself)
        for j in 0..n_units {
            if j != target_unit {
                let dist = compute_unit_distance_for_obs(y, d, j, target_unit, target_period);
                if dist.is_finite() {
                    unit_weights[j] = (-lambda_unit * dist).exp();
                }
                // Units with infinite distance (no valid comparison periods) get weight 0
            }
        }
    }

    // Target unit gets weight 1 (will be masked out in estimation anyway)
    unit_weights[target_unit] = 1.0;

    // Outer product: W[t, i] = θ_s × ω_j (raw exponentials, unnormalized)
    let mut weight_matrix = Array2::<f64>::zeros((n_periods, n_units));
    for t in 0..n_periods {
        for i in 0..n_units {
            weight_matrix[[t, i]] = time_weights[t] * unit_weights[i];
        }
    }

    weight_matrix
}

/// Estimate TROP model using alternating minimization.
///
/// Minimizes: Σ W_{ti}(Y_{ti} - α_i - β_t - L_{ti})² + λ_nn||L||_*
///
/// Paper alignment: Uses weighted proximal gradient for L update with
/// Lipschitz constant L_f = 2·max(W), step size η = 1/(2·max(W)):
///   G = L + (W/max(W)) ⊙ (R - L)
///   L ← prox_{η·λ_nn·||·||_*}(G)
///
/// Returns None if estimation fails due to numerical issues.
#[allow(clippy::too_many_arguments)]
fn estimate_model(
    y: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    weight_matrix: &ArrayView2<f64>,
    lambda_nn: f64,
    n_periods: usize,
    n_units: usize,
    max_iter: usize,
    tol: f64,
    exclude_obs: Option<(usize, usize)>,
) -> Option<(Array1<f64>, Array1<f64>, Array2<f64>)> {
    // Create estimation mask
    let mut est_mask =
        Array2::<bool>::from_shape_fn((n_periods, n_units), |(t, i)| control_mask[[t, i]] != 0);

    if let Some((t_ex, i_ex)) = exclude_obs {
        est_mask[[t_ex, i_ex]] = false;
    }

    // Valid mask: non-NaN and in estimation set
    let valid_mask = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
        y[[t, i]].is_finite() && est_mask[[t, i]]
    });

    // Masked weights: W=0 for invalid/treated observations
    let w_masked = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
        if valid_mask[[t, i]] {
            weight_matrix[[t, i]]
        } else {
            0.0
        }
    });

    // Lipschitz constant of ∇f is L_f = 2·max(W), so prox threshold = λ/(2·max(W))
    let w_max = w_masked.iter().cloned().fold(0.0_f64, f64::max);
    let prox_threshold = if w_max > 0.0 {
        lambda_nn / (2.0 * w_max)
    } else {
        lambda_nn / 2.0
    };

    // Weight sums per unit and time
    let weight_sum_per_unit: Array1<f64> = w_masked.sum_axis(Axis(0));
    let weight_sum_per_time: Array1<f64> = w_masked.sum_axis(Axis(1));

    // Safe denominators
    let safe_unit_denom: Array1<f64> = weight_sum_per_unit.mapv(|w| if w > 0.0 { w } else { 1.0 });
    let safe_time_denom: Array1<f64> = weight_sum_per_time.mapv(|w| if w > 0.0 { w } else { 1.0 });

    let unit_has_obs: Array1<bool> = weight_sum_per_unit.mapv(|w| w > 0.0);
    let time_has_obs: Array1<bool> = weight_sum_per_time.mapv(|w| w > 0.0);

    // Safe Y (replace NaN with 0)
    let y_safe = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
        if y[[t, i]].is_finite() {
            y[[t, i]]
        } else {
            0.0
        }
    });

    // Initialize
    let mut alpha = Array1::<f64>::zeros(n_units);
    let mut beta = Array1::<f64>::zeros(n_periods);
    let mut l = Array2::<f64>::zeros((n_periods, n_units));

    // Alternating minimization
    for _ in 0..max_iter {
        let alpha_old = alpha.clone();
        let beta_old = beta.clone();
        let l_old = l.clone();

        // Step 1: Update α and β (weighted least squares)
        // R = Y - L
        let r = &y_safe - &l;

        // Alpha update: α_i = Σ_t W_{ti}(R_{ti} - β_t) / Σ_t W_{ti}
        for i in 0..n_units {
            if unit_has_obs[i] {
                let mut num = 0.0;
                for t in 0..n_periods {
                    num += w_masked[[t, i]] * (r[[t, i]] - beta[t]);
                }
                alpha[i] = num / safe_unit_denom[i];
            }
        }

        // Beta update: β_t = Σ_i W_{ti}(R_{ti} - α_i) / Σ_i W_{ti}
        for t in 0..n_periods {
            if time_has_obs[t] {
                let mut num = 0.0;
                for i in 0..n_units {
                    num += w_masked[[t, i]] * (r[[t, i]] - alpha[i]);
                }
                beta[t] = num / safe_time_denom[t];
            }
        }

        // Step 2: Update L with WEIGHTED nuclear norm penalty
        // Inner FISTA-accelerated proximal gradient loop (α, β fixed)
        // L ← prox_{threshold·||·||_*}(L + W_norm ⊙ (R - L))
        // where R = Y - α - β, W_norm = W/max(W)

        // Compute target residual R = Y - α - β
        let mut r_target = Array2::<f64>::zeros((n_periods, n_units));
        for t in 0..n_periods {
            for i in 0..n_units {
                r_target[[t, i]] = y_safe[[t, i]] - alpha[i] - beta[t];
            }
        }

        // For W=0 cells, use current L instead of R (prevent absorbing treatment)
        let r_masked = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
            if w_masked[[t, i]] > 0.0 {
                r_target[[t, i]]
            } else {
                l[[t, i]]
            }
        });

        // Normalize weights: W_norm = W / W_max (max becomes 1)
        let w_norm = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
            if w_max > 0.0 {
                w_masked[[t, i]] / w_max
            } else {
                w_masked[[t, i]]
            }
        });

        // FISTA inner loop for L update
        let mut l_prev = l.clone();
        let mut t_fista = 1.0_f64;
        let max_inner_iter = 10;

        for _ in 0..max_inner_iter {
            let l_inner_old = l.clone();

            // FISTA momentum
            let t_fista_new = (1.0 + (1.0 + 4.0 * t_fista * t_fista).sqrt()) / 2.0;
            let momentum = (t_fista - 1.0) / t_fista_new;
            let l_momentum = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
                l[[t, i]] + momentum * (l[[t, i]] - l_prev[[t, i]])
            });

            // Gradient step from momentum point
            let mut gradient_step = Array2::<f64>::zeros((n_periods, n_units));
            for t in 0..n_periods {
                for i in 0..n_units {
                    gradient_step[[t, i]] = l_momentum[[t, i]]
                        + w_norm[[t, i]] * (r_masked[[t, i]] - l_momentum[[t, i]]);
                }
            }

            // Proximal step: soft-threshold with corrected threshold
            l_prev = l.clone();
            l = soft_threshold_svd(&gradient_step, prox_threshold)?;
            t_fista = t_fista_new;

            // Check inner convergence
            let l_inner_diff = max_abs_diff_2d(&l, &l_inner_old);
            if l_inner_diff < tol {
                break;
            }
        }

        // Check convergence
        let alpha_diff = max_abs_diff(&alpha, &alpha_old);
        let beta_diff = max_abs_diff(&beta, &beta_old);
        let l_diff = max_abs_diff_2d(&l, &l_old);

        if alpha_diff.max(beta_diff).max(l_diff) < tol {
            break;
        }
    }

    Some((alpha, beta, l))
}

/// Apply soft-thresholding to singular values (proximal operator for nuclear norm).
fn soft_threshold_svd(m: &Array2<f64>, threshold: f64) -> Option<Array2<f64>> {
    if threshold <= 0.0 {
        return Some(m.clone());
    }

    // Check for non-finite values
    if !m.iter().all(|&x| x.is_finite()) {
        return Some(Array2::zeros(m.raw_dim()));
    }

    let n_rows = m.nrows();
    let n_cols = m.ncols();

    // Convert ndarray to faer using Mat::from_fn (pure Rust, no external deps)
    let m_faer = faer::Mat::from_fn(n_rows, n_cols, |i, j| m[[i, j]]);

    // Compute thin SVD using faer (capitalized methods in faer 0.24)
    let svd = match m_faer.thin_svd() {
        Ok(s) => s,
        Err(_) => return Some(Array2::zeros(m.raw_dim())),
    };

    let u_faer = svd.U();
    let s_diag = svd.S(); // Returns diagonal view
    let s_col = s_diag.column_vector(); // Get as column vector
    let v_faer = svd.V(); // This is V, not V^T

    let s_len = s_col.nrows();

    // Check for non-finite SVD output
    for i in 0..u_faer.nrows() {
        for j in 0..u_faer.ncols() {
            if !u_faer[(i, j)].is_finite() {
                return Some(Array2::zeros(m.raw_dim()));
            }
        }
    }
    for i in 0..s_len {
        if !s_col[i].is_finite() {
            return Some(Array2::zeros(m.raw_dim()));
        }
    }
    for i in 0..v_faer.nrows() {
        for j in 0..v_faer.ncols() {
            if !v_faer[(i, j)].is_finite() {
                return Some(Array2::zeros(m.raw_dim()));
            }
        }
    }

    // Soft-threshold singular values and count non-zero
    let mut s_thresh = Vec::with_capacity(s_len);
    let mut nonzero_count = 0;
    for i in 0..s_len {
        let sv = s_col[i];
        let sv_thresh = (sv - threshold).max(0.0);
        s_thresh.push(sv_thresh);
        if sv_thresh > 1e-10 {
            nonzero_count += 1;
        }
    }

    if nonzero_count == 0 {
        return Some(Array2::zeros(m.raw_dim()));
    }

    // Truncated reconstruction: U @ diag(s_thresh) @ V^T
    let mut result = Array2::<f64>::zeros((n_rows, n_cols));

    for k in 0..s_thresh.len() {
        if s_thresh[k] > 1e-10 {
            for i in 0..n_rows {
                for j in 0..n_cols {
                    // u_faer[(i, k)] * s_thresh[k] * v_faer[(j, k)] (since V^T[k,j] = V[j,k])
                    result[[i, j]] += s_thresh[k] * u_faer[(i, k)] * v_faer[(j, k)];
                }
            }
        }
    }

    Some(result)
}

/// Maximum absolute difference between two 1D arrays.
#[inline]
fn max_abs_diff(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Maximum absolute difference between two 2D arrays.
#[inline]
fn max_abs_diff_2d(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Compute bootstrap variance estimation for TROP in parallel (local method).
///
/// Performs unit-level block bootstrap, parallelizing across bootstrap iterations.
///
/// # Arguments
/// * `y` - Outcome matrix (n_periods x n_units)
/// * `d` - Treatment indicator matrix (n_periods x n_units)
/// * `control_mask` - Boolean mask for control observations
/// * `time_dist_matrix` - Pre-computed time distance matrix
/// * `lambda_time` - Selected time decay parameter
/// * `lambda_unit` - Selected unit distance parameter
/// * `lambda_nn` - Selected nuclear norm parameter
/// * `n_bootstrap` - Number of bootstrap iterations
/// * `max_iter` - Maximum iterations for model estimation
/// * `tol` - Convergence tolerance
/// * `control_indices` - Pre-generated stratified bootstrap indices for the
///   control pool, shape `(n_bootstrap, n_control_units)`, dtype `i64`.
///   Values must be in `[0, n_control_units)`.
/// * `treated_indices` - Pre-generated stratified bootstrap indices for the
///   treated pool, shape `(n_bootstrap, n_treated_units)`, dtype `i64`.
///   Values must be in `[0, n_treated_units)`.
/// * `survey_weights` - Optional unit-level survey weights (length n_units).
///   When provided, ATT is computed as a weighted mean of per-observation
///   treatment effects using unit weights. Model fitting, LOOCV, and distance
///   computation are unchanged.
///
/// The index arrays carry the RNG contract: they are produced on the Python
/// side by `diff_diff.bootstrap_utils.stratified_bootstrap_indices` with a
/// numpy `default_rng(seed)`, so Rust and Python consumers see identical
/// sampling under the same seed. Invalid index values (negative or out of
/// range) raise a `PyValueError` rather than silently producing malformed
/// bootstrap samples.
///
/// # Returns
/// (bootstrap_estimates, standard_error)
#[pyfunction]
#[pyo3(signature = (y, d, control_mask, time_dist_matrix, lambda_time, lambda_unit, lambda_nn, n_bootstrap, max_iter, tol, control_indices, treated_indices, survey_weights=None))]
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_trop_variance<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    d: PyReadonlyArray2<'py, f64>,
    control_mask: PyReadonlyArray2<'py, u8>,
    time_dist_matrix: PyReadonlyArray2<'py, i64>,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: usize,
    max_iter: usize,
    tol: f64,
    control_indices: PyReadonlyArray2<'py, i64>,
    treated_indices: PyReadonlyArray2<'py, i64>,
    survey_weights: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let y_arr = y.as_array().to_owned();
    let d_arr = d.as_array().to_owned();
    let control_mask_arr = control_mask.as_array().to_owned();
    let time_dist_arr = time_dist_matrix.as_array().to_owned();
    let ctrl_idx_arr = control_indices.as_array().to_owned();
    let trt_idx_arr = treated_indices.as_array().to_owned();
    let sw_arr: Option<Array1<f64>> = survey_weights.map(|sw| sw.as_array().to_owned());

    let n_units = y_arr.ncols();
    let n_periods = y_arr.nrows();

    // Issue D fix: Identify treated and control units for stratified sampling
    // Following paper's Algorithm 3 (page 27): sample N_0 control and N_1 treated separately
    let mut original_treated_units: Vec<usize> = Vec::new();
    let mut original_control_units: Vec<usize> = Vec::new();
    for i in 0..n_units {
        let is_ever_treated = (0..n_periods).any(|t| d_arr[[t, i]] == 1.0);
        if is_ever_treated {
            original_treated_units.push(i);
        } else {
            original_control_units.push(i);
        }
    }
    let n_treated_units = original_treated_units.len();
    let n_control_units = original_control_units.len();

    // Validate index-array shapes match the stratified pool sizes
    if ctrl_idx_arr.shape() != [n_bootstrap, n_control_units] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "control_indices shape {:?} does not match (n_bootstrap={}, n_control_units={})",
            ctrl_idx_arr.shape(),
            n_bootstrap,
            n_control_units,
        )));
    }
    if trt_idx_arr.shape() != [n_bootstrap, n_treated_units] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "treated_indices shape {:?} does not match (n_bootstrap={}, n_treated_units={})",
            trt_idx_arr.shape(),
            n_bootstrap,
            n_treated_units,
        )));
    }

    // Validate index values are in range. Fail fast with a clean PyValueError
    // rather than panicking inside the parallel loop on a negative cast or an
    // out-of-pool Vec index.
    if n_control_units > 0 {
        let n_ctrl = n_control_units as i64;
        for v in ctrl_idx_arr.iter() {
            if *v < 0 || *v >= n_ctrl {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "control_indices contains out-of-range value {} (valid: [0, {}))",
                    v, n_control_units,
                )));
            }
        }
    }
    if n_treated_units > 0 {
        let n_trt = n_treated_units as i64;
        for v in trt_idx_arr.iter() {
            if *v < 0 || *v >= n_trt {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "treated_indices contains out-of-range value {} (valid: [0, {}))",
                    v, n_treated_units,
                )));
            }
        }
    }

    // Run bootstrap iterations in parallel
    // RNG-canonical contract: control_indices and treated_indices are pre-generated
    // by numpy.random.default_rng(seed) on the Python side via
    // diff_diff.bootstrap_utils.stratified_bootstrap_indices, so SE is identical
    // across backends under the same seed (silent-failures finding #23).
    let bootstrap_estimates: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .filter_map(|b| {
            // Stratified sampling: consume pre-generated indices for replicate b
            let mut sampled_units: Vec<usize> = Vec::with_capacity(n_units);
            for j in 0..n_control_units {
                let idx = ctrl_idx_arr[[b, j]] as usize;
                sampled_units.push(original_control_units[idx]);
            }
            for j in 0..n_treated_units {
                let idx = trt_idx_arr[[b, j]] as usize;
                sampled_units.push(original_treated_units[idx]);
            }

            // Create bootstrap matrices by selecting columns
            let mut y_boot = Array2::<f64>::zeros((n_periods, n_units));
            let mut d_boot = Array2::<f64>::zeros((n_periods, n_units));
            let mut control_mask_boot = Array2::<u8>::zeros((n_periods, n_units));

            for (new_idx, &old_idx) in sampled_units.iter().enumerate() {
                for t in 0..n_periods {
                    y_boot[[t, new_idx]] = y_arr[[t, old_idx]];
                    d_boot[[t, new_idx]] = d_arr[[t, old_idx]];
                    control_mask_boot[[t, new_idx]] = control_mask_arr[[t, old_idx]];
                }
            }

            // Get treated observations in bootstrap sample
            let mut boot_treated: Vec<(usize, usize)> = Vec::new();
            for t in 0..n_periods {
                for i in 0..n_units {
                    if d_boot[[t, i]] == 1.0 {
                        boot_treated.push((t, i));
                    }
                }
            }

            if boot_treated.is_empty() {
                return None;
            }

            // Get control units in bootstrap sample (units never treated)
            let mut boot_control_units: Vec<usize> = Vec::new();
            for i in 0..n_units {
                let is_control = (0..n_periods).all(|t| d_boot[[t, i]] == 0.0);
                if is_control {
                    boot_control_units.push(i);
                }
            }

            if boot_control_units.is_empty() {
                return None;
            }

            // Compute ATT for bootstrap sample
            // When survey weights are provided, ATT is a weighted mean of
            // per-observation treatment effects using unit-level weights.
            let mut tau_sum = 0.0;
            let mut weight_sum = 0.0;
            let mut tau_count = 0usize;

            for (t, i) in boot_treated {
                // Skip non-finite outcomes (match main fit NaN contract)
                if !y_boot[[t, i]].is_finite() {
                    continue;
                }

                let weight_matrix = compute_weight_matrix(
                    &y_boot.view(),
                    &d_boot.view(),
                    n_periods,
                    n_units,
                    i,
                    t,
                    lambda_time,
                    lambda_unit,
                    &time_dist_arr.view(),
                );

                if let Some((alpha, beta, l)) = estimate_model(
                    &y_boot.view(),
                    &control_mask_boot.view(),
                    &weight_matrix.view(),
                    lambda_nn,
                    n_periods,
                    n_units,
                    max_iter,
                    tol,
                    None,
                ) {
                    let tau = y_boot[[t, i]] - alpha[i] - beta[t] - l[[t, i]];
                    let w = match &sw_arr {
                        Some(sw) => sw[sampled_units[i]],
                        None => 1.0,
                    };
                    tau_sum += w * tau;
                    weight_sum += w;
                    tau_count += 1;
                }
            }

            if tau_count == 0 {
                None
            } else {
                Some(tau_sum / weight_sum)
            }
        })
        .collect();

    // Compute standard error
    // Return NaN when < 2 samples to properly propagate undefined inference
    let se = if bootstrap_estimates.len() < 2 {
        f64::NAN
    } else {
        let n = bootstrap_estimates.len() as f64;
        let mean = bootstrap_estimates.iter().sum::<f64>() / n;
        let variance = bootstrap_estimates
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        variance.sqrt()
    };

    let estimates_arr = Array1::from_vec(bootstrap_estimates);
    Ok((estimates_arr.to_pyarray(py), se))
}

// ============================================================================
// Global method implementation
//
// Note: Only the #[pyfunction] exports were renamed (joint → global) to match
// the Python public API. The private Rust helpers below retain their original
// `*_joint*` names to keep the Rust-only rename scope minimal.
// ============================================================================

/// Compute global weights for global method estimation.
///
/// Unlike local (which computes per-observation weights), global uses
/// weights based on:
/// - Time weights: distance to center of treated block
/// - Unit weights: RMSE to average treated trajectory over pre-periods
///
/// # Arguments
/// * `y` - Outcome matrix (n_periods x n_units)
/// * `d` - Treatment indicator matrix (n_periods x n_units)
/// * `lambda_time` - Time weight decay parameter
/// * `lambda_unit` - Unit weight decay parameter
/// * `treated_periods` - Number of post-treatment periods
///
/// # Returns
/// Weight matrix (n_periods x n_units)
fn compute_joint_weights(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    lambda_time: f64,
    lambda_unit: f64,
    treated_periods: usize,
) -> Array2<f64> {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    // Identify treated units (ever treated)
    let mut treated_unit_idx: Vec<usize> = Vec::new();
    for i in 0..n_units {
        if (0..n_periods).any(|t| d[[t, i]] == 1.0) {
            treated_unit_idx.push(i);
        }
    }

    // Time weights: distance to center of treated block
    // center = T - treated_periods / 2
    let center = n_periods as f64 - treated_periods as f64 / 2.0;
    let mut delta_time = Array1::<f64>::zeros(n_periods);
    for t in 0..n_periods {
        let dist = (t as f64 - center).abs();
        delta_time[t] = (-lambda_time * dist).exp();
    }

    // Unit weights: RMSE to average treated trajectory over pre-periods
    let n_pre = n_periods.saturating_sub(treated_periods);

    // Compute average treated trajectory
    // Initialize to NaN so periods with all-NaN treated data stay NaN (excluded from RMSE)
    let mut average_treated = Array1::<f64>::from_elem(n_periods, f64::NAN);
    if !treated_unit_idx.is_empty() {
        for t in 0..n_periods {
            let mut sum = 0.0;
            let mut count = 0;
            for &i in &treated_unit_idx {
                if y[[t, i]].is_finite() {
                    sum += y[[t, i]];
                    count += 1;
                }
            }
            if count > 0 {
                average_treated[t] = sum / count as f64;
            }
            // If count == 0, average_treated[t] stays NaN (correctly excluded)
        }
    }

    // Compute RMS distance for each unit over pre-periods
    let mut delta_unit = Array1::<f64>::zeros(n_units);
    for i in 0..n_units {
        if n_pre > 0 {
            let mut sum_sq = 0.0;
            let mut n_valid = 0;
            for t in 0..n_pre {
                if y[[t, i]].is_finite() && average_treated[t].is_finite() {
                    let diff = average_treated[t] - y[[t, i]];
                    sum_sq += diff * diff;
                    n_valid += 1;
                }
            }
            let dist = if n_valid > 0 {
                (sum_sq / n_valid as f64).sqrt()
            } else {
                // No valid pre-period observations for this unit.
                // Set dist = infinity so delta_unit = exp(-infinity) = 0.
                // This ensures units with no valid pre-period data get zero weight,
                // matching the Python behavior.
                f64::INFINITY
            };
            delta_unit[i] = (-lambda_unit * dist).exp();
        } else {
            delta_unit[i] = 1.0;
        }
    }

    // Outer product: (n_periods x n_units)
    let mut delta = Array2::<f64>::zeros((n_periods, n_units));
    for t in 0..n_periods {
        for i in 0..n_units {
            delta[[t, i]] = delta_time[t] * delta_unit[i];
        }
    }

    // (1-W) masking: zero out treated observations per paper Eq. 2
    for t in 0..n_periods {
        for i in 0..n_units {
            delta[[t, i]] *= 1.0 - d[[t, i]];
        }
    }

    delta
}

/// Solve global TWFE via weighted least squares (no low-rank, no tau).
///
/// Minimizes: min Σ δ_{it}(Y_{it} - μ - α_i - β_t)²
///
/// tau is extracted post-hoc by the caller (ATT = mean residual over treated).
///
/// # Returns
/// (mu, alpha, beta) estimated parameters
fn solve_joint_no_lowrank(
    y: &ArrayView2<f64>,
    delta: &ArrayView2<f64>,
) -> Option<(f64, Array1<f64>, Array1<f64>)> {
    // SVD-based minimum-norm weighted least-squares fit — mirrors Python's
    // `_solve_global_no_lowrank` at `diff_diff/trop_global.py:340-412`
    // step-for-step so Rust and Python paths produce the same canonical
    // solution on rank-deficient Y (silent-failures finding #23).
    //
    // Model: Y_it = μ + α_i + β_t + ε_it, with α_0 = β_0 = 0 for
    // identification. Weights: δ_it. Flatten row-major with
    // idx = t * n_units + i (matches Python's Y.flatten() C-order).
    let n_periods = y.nrows();
    let n_units = y.ncols();
    let n_obs = n_periods * n_units;
    let n_params = 1 + (n_units - 1) + (n_periods - 1);

    // Flatten y + weights with NaN masking — matches trop_global.py:354-360.
    let mut y_flat = Array1::<f64>::zeros(n_obs);
    let mut w_flat = Array1::<f64>::zeros(n_obs);
    let mut sum_w = 0.0;
    for t in 0..n_periods {
        for i in 0..n_units {
            let idx = t * n_units + i;
            let y_ti = y[[t, i]];
            let w_ti = delta[[t, i]];
            let valid = y_ti.is_finite() && w_ti.is_finite();
            let w = if valid { w_ti.max(0.0) } else { 0.0 };
            y_flat[idx] = if valid { y_ti } else { 0.0 };
            w_flat[idx] = w;
            sum_w += w;
        }
    }

    // All-zero weights short-circuit — matches trop_global.py:366.
    if sum_w < 1e-10 {
        return None;
    }

    // Build design matrix X = [intercept | unit_dummies[1..] | time_dummies[1..]]
    // — matches trop_global.py:374-385. Explicit nested loops so the
    // index correspondence with Python is unambiguous.
    let mut x = Array2::<f64>::zeros((n_obs, n_params));
    for t in 0..n_periods {
        for i in 0..n_units {
            let idx = t * n_units + i;
            x[[idx, 0]] = 1.0; // intercept
            if i >= 1 {
                x[[idx, i]] = 1.0; // unit dummy (unit 0 dropped)
            }
            if t >= 1 {
                x[[idx, (n_units - 1) + t]] = 1.0; // time dummy (period 0 dropped)
            }
        }
    }

    // Apply sqrt-weights: X_w = X * sqrt(w)[:, None], y_w = y * sqrt(w).
    // Matches trop_global.py:387-389.
    let sqrt_w: Array1<f64> = w_flat.mapv(|w| w.sqrt());
    for r in 0..n_obs {
        let s = sqrt_w[r];
        for c in 0..n_params {
            x[[r, c]] *= s;
        }
        y_flat[r] *= s;
    }

    // Solve via SVD with numpy-compatible rcond truncation.
    let coeffs = solve_wls_svd(&x.view(), &y_flat.view())?;

    // Unpack: mu = coeffs[0], alpha[1..] = coeffs[1..n_units],
    // beta[1..] = coeffs[n_units..]. Matches trop_global.py:406-410.
    let mu = coeffs[0];
    let mut alpha = Array1::<f64>::zeros(n_units);
    for i in 1..n_units {
        alpha[i] = coeffs[i];
    }
    let mut beta = Array1::<f64>::zeros(n_periods);
    for t in 1..n_periods {
        beta[t] = coeffs[(n_units - 1) + t];
    }

    Some((mu, alpha, beta))
}

/// Minimum-norm least-squares solution via faer thin SVD with rcond truncation.
///
/// Mirrors `np.linalg.lstsq(A, b, rcond=None)` from numpy: singular values
/// below `rcond * max(S)` with `rcond = eps * max(n_rows, n_cols)` are
/// treated as zero. On rank-deficient A this returns the unique
/// minimum-norm least-squares solution.
///
/// This helper intentionally does NOT reuse `rust/src/linalg.rs::solve_ols`
/// because `solve_ols` hard-codes `rcond = 1e-7` (R's `lm()` default) which
/// would truncate singular values that numpy's default keeps. TROP's
/// canonical Python path is numpy-compatible; Rust must match.
///
/// Returns `None` only when the SVD itself fails (rare on finite inputs);
/// the caller (LOOCV / FISTA / bootstrap) interprets `None` as an
/// unsuccessful fit.
fn solve_wls_svd(a: &ArrayView2<f64>, b: &ArrayView1<f64>) -> Option<Array1<f64>> {
    let n_rows = a.nrows();
    let n_cols = a.ncols();
    let a_owned = a.to_owned();
    let b_owned = b.to_owned();

    // Convert ndarray -> faer for SVD.
    let a_faer = ndarray_to_faer(&a_owned);

    let svd = a_faer.thin_svd().ok()?;

    let u_faer = svd.U();
    let s_diag = svd.S();
    let s_col = s_diag.column_vector();
    let v_faer = svd.V();

    // Extract U (n_rows x min(n,k)) back to ndarray.
    let u_rows = u_faer.nrows();
    let u_cols = u_faer.ncols();
    let mut u = Array2::<f64>::zeros((u_rows, u_cols));
    for i in 0..u_rows {
        for j in 0..u_cols {
            u[[i, j]] = u_faer[(i, j)];
        }
    }

    // Extract singular values.
    let s_len = s_col.nrows();
    let mut s = Array1::<f64>::zeros(s_len);
    for i in 0..s_len {
        s[i] = s_col[i];
    }

    // Extract V (k x min(n,k)) back to ndarray. faer's V is not V^T.
    let v_rows = v_faer.nrows();
    let v_cols = v_faer.ncols();
    let mut v = Array2::<f64>::zeros((v_rows, v_cols));
    for i in 0..v_rows {
        for j in 0..v_cols {
            v[[i, j]] = v_faer[(i, j)];
        }
    }

    // numpy rcond = eps * max(n_rows, n_cols); truncate s[i] <= rcond * max(s).
    let rcond = f64::EPSILON * (n_rows.max(n_cols) as f64);
    let s_max = s.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = s_max * rcond;

    // Compute β = V * S^{-1}_truncated * U^T * y.
    let uty = u.t().dot(&b_owned); // (min(n,k),)
    let mut s_inv_uty = Array1::<f64>::zeros(s_len);
    for i in 0..s_len {
        if s[i] > threshold {
            s_inv_uty[i] = uty[i] / s[i];
        }
        // else: leave 0 — this is the pseudo-inverse / minimum-norm step
        // that also covers Python's `except LinAlgError: pinv(...)` fallback
        // tier, since faer thin_svd is numerically robust on finite inputs.
    }
    let coeffs = v.dot(&s_inv_uty);

    Some(coeffs)
}

/// Solve global TWFE + low-rank via alternating minimization (no tau).
///
/// Minimizes: min Σ δ_{it}(Y_{it} - μ - α_i - β_t - L_{it})² + λ_nn||L||_*
///
/// tau is extracted post-hoc by the caller (ATT = mean residual over treated).
///
/// # Returns
/// (mu, alpha, beta, L) estimated parameters
fn solve_joint_with_lowrank(
    y: &ArrayView2<f64>,
    delta: &ArrayView2<f64>,
    lambda_nn: f64,
    max_iter: usize,
    tol: f64,
) -> Option<(f64, Array1<f64>, Array1<f64>, Array2<f64>)> {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    // Precompute normalized weights and threshold (constant across iterations)
    let delta_max = delta.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = if delta_max > 0.0 {
        lambda_nn / (2.0 * delta_max)
    } else {
        lambda_nn / 2.0
    };

    // Precompute delta_norm (masked for NaN outcomes)
    let mut delta_norm = Array2::<f64>::zeros((n_periods, n_units));
    for t in 0..n_periods {
        for i in 0..n_units {
            let d_ti = if y[[t, i]].is_finite() {
                delta[[t, i]]
            } else {
                0.0
            };
            delta_norm[[t, i]] = if delta_max > 0.0 {
                d_ti / delta_max
            } else {
                d_ti
            };
        }
    }

    // Initialize L = 0
    let mut l = Array2::<f64>::zeros((n_periods, n_units));

    for _ in 0..max_iter {
        let l_old = l.clone();

        // Step 1: Fix L, solve for (mu, alpha, beta)
        let y_adj = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
            y[[t, i]] - l[[t, i]] // NaN - finite = NaN (preserves NaN info)
        });
        let (mu, alpha, beta) = solve_joint_no_lowrank(&y_adj.view(), delta)?;

        // Step 2: Fix (mu, alpha, beta), update L with FISTA acceleration
        // Residual: R = Y - mu - alpha - beta
        // For delta=0 observations (treated/NaN), keep L rather than R
        let mut r_masked = Array2::<f64>::zeros((n_periods, n_units));
        for t in 0..n_periods {
            for i in 0..n_units {
                if delta_norm[[t, i]] > 0.0 && y[[t, i]].is_finite() {
                    r_masked[[t, i]] = y[[t, i]] - mu - alpha[i] - beta[t];
                } else {
                    r_masked[[t, i]] = l[[t, i]];
                }
            }
        }

        // Inner FISTA loop for L update
        let mut l_inner = l.clone();
        let mut l_inner_prev = l_inner.clone();
        let mut t_fista = 1.0_f64;

        for _ in 0..20 {
            // FISTA momentum
            let t_fista_new = (1.0 + (1.0 + 4.0 * t_fista * t_fista).sqrt()) / 2.0;
            let momentum = (t_fista - 1.0) / t_fista_new;

            // Gradient step from momentum point
            let mut gradient_step = Array2::<f64>::zeros((n_periods, n_units));
            for t in 0..n_periods {
                for i in 0..n_units {
                    let l_mom =
                        l_inner[[t, i]] + momentum * (l_inner[[t, i]] - l_inner_prev[[t, i]]);
                    gradient_step[[t, i]] = l_mom + delta_norm[[t, i]] * (r_masked[[t, i]] - l_mom);
                }
            }

            // Proximal step: soft-threshold singular values
            // l_inner_prev holds pre-SVD value for both momentum and convergence check
            l_inner_prev = l_inner;
            l_inner = soft_threshold_svd(&gradient_step, threshold)?;
            t_fista = t_fista_new;

            // Convergence check
            let inner_diff = max_abs_diff_2d(&l_inner, &l_inner_prev);
            if inner_diff < tol {
                break;
            }
        }

        l = l_inner;

        // Outer convergence check
        let l_diff = max_abs_diff_2d(&l, &l_old);
        if l_diff < tol {
            break;
        }
    }

    // Final solve with converged L
    let y_adj = Array2::from_shape_fn((n_periods, n_units), |(t, i)| y[[t, i]] - l[[t, i]]);
    let (mu, alpha, beta) = solve_joint_no_lowrank(&y_adj.view(), delta)?;

    Some((mu, alpha, beta, l))
}

/// Compute LOOCV score for global method with specific parameter combination.
///
/// Following paper's Equation 5:
/// Q(λ) = Σ_{j,s: D_js=0} [τ̂_js^loocv(λ)]²
///
/// For global method, we exclude each control observation, fit the global model
/// on remaining data, and compute the pseudo-treatment effect at the excluded obs.
///
/// # Returns
/// (score, n_valid, first_failed_obs)
#[allow(clippy::too_many_arguments)]
fn loocv_score_joint(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_obs: &[(usize, usize)],
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    treated_periods: usize,
    max_iter: usize,
    tol: f64,
) -> (f64, usize, Option<(usize, usize)>) {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    // Compute global weights (same for all LOOCV iterations)
    let delta = compute_joint_weights(y, d, lambda_time, lambda_unit, treated_periods);

    // Parallelize over control observations — each per-observation computation
    // is independent (clone delta, zero one entry, fit model, extract τ²).
    // with_min_len(64) prevents scheduling overhead from dominating on small panels.
    let (tau_sq_sum, n_valid, first_failed) = control_obs
        .par_iter()
        .with_min_len(64)
        .fold(
            || (0.0f64, 0usize, None::<(usize, usize)>),
            |(sum, valid, first_fail), &(t_ex, i_ex)| {
                let mut delta_ex = delta.clone();
                delta_ex[[t_ex, i_ex]] = 0.0;

                let result = if lambda_nn >= 1e10 {
                    solve_joint_no_lowrank(y, &delta_ex.view()).map(|(mu, alpha, beta)| {
                        let l = Array2::<f64>::zeros((n_periods, n_units));
                        (mu, alpha, beta, l)
                    })
                } else {
                    solve_joint_with_lowrank(y, &delta_ex.view(), lambda_nn, max_iter, tol)
                };

                match result {
                    Some((mu, alpha, beta, l)) => {
                        if y[[t_ex, i_ex]].is_finite() {
                            let tau_loocv =
                                y[[t_ex, i_ex]] - mu - alpha[i_ex] - beta[t_ex] - l[[t_ex, i_ex]];
                            (sum + tau_loocv * tau_loocv, valid + 1, first_fail)
                        } else {
                            (sum, valid, first_fail)
                        }
                    }
                    None => (sum, valid, first_fail.or(Some((t_ex, i_ex)))),
                }
            },
        )
        .reduce(
            || (0.0, 0, None),
            |(s1, v1, f1), (s2, v2, f2)| (s1 + s2, v1 + v2, f1.or(f2)),
        );

    // Per Equation 5: if ANY fit fails, this λ combination is invalid
    if first_failed.is_some() {
        return (f64::INFINITY, n_valid, first_failed);
    }

    if n_valid == 0 {
        (f64::INFINITY, 0, None)
    } else {
        (tau_sq_sum, n_valid, None)
    }
}

/// Perform LOOCV grid search for global method using parallel grid search.
///
/// Evaluates all combinations of (lambda_time, lambda_unit, lambda_nn) in parallel
/// and returns the combination with lowest LOOCV score.
///
/// # Arguments
/// * `y` - Outcome matrix (n_periods x n_units)
/// * `d` - Treatment indicator matrix (n_periods x n_units)
/// * `control_mask` - Boolean mask (n_periods x n_units) for control observations
/// * `lambda_time_grid` - Grid of time decay parameters
/// * `lambda_unit_grid` - Grid of unit distance parameters
/// * `lambda_nn_grid` - Grid of nuclear norm parameters
/// * `max_iter` - Maximum iterations for model estimation
/// * `tol` - Convergence tolerance
///
/// # Returns
/// (best_lambda_time, best_lambda_unit, best_lambda_nn, best_score, n_valid, n_attempted, first_failed_obs)
#[pyfunction]
#[pyo3(signature = (y, d, control_mask, lambda_time_grid, lambda_unit_grid, lambda_nn_grid, max_iter, tol))]
#[allow(clippy::too_many_arguments)]
pub fn loocv_grid_search_global<'py>(
    _py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    d: PyReadonlyArray2<'py, f64>,
    control_mask: PyReadonlyArray2<'py, u8>,
    lambda_time_grid: PyReadonlyArray1<'py, f64>,
    lambda_unit_grid: PyReadonlyArray1<'py, f64>,
    lambda_nn_grid: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<(f64, f64, f64, f64, usize, usize, Option<(usize, usize)>)> {
    let y_arr = y.as_array();
    let d_arr = d.as_array();
    let control_mask_arr = control_mask.as_array();
    let lambda_time_vec: Vec<f64> = lambda_time_grid.as_array().to_vec();
    let lambda_unit_vec: Vec<f64> = lambda_unit_grid.as_array().to_vec();
    let lambda_nn_vec: Vec<f64> = lambda_nn_grid.as_array().to_vec();

    // Validate: lambda_time_grid and lambda_unit_grid must not contain inf.
    // Per Athey et al. (2025) Eq. 3: use 0.0 for uniform weights, not inf.
    for &v in &lambda_time_vec {
        if v.is_infinite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "lambda_time_grid must not contain inf. Use 0.0 for uniform weights (disabled) per Athey et al. (2025) Eq. 3."
            ));
        }
    }
    for &v in &lambda_unit_vec {
        if v.is_infinite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "lambda_unit_grid must not contain inf. Use 0.0 for uniform weights (disabled) per Athey et al. (2025) Eq. 3."
            ));
        }
    }

    let n_periods = y_arr.nrows();
    let n_units = y_arr.ncols();

    // Determine treated periods from D matrix
    let mut first_treat_period = n_periods;
    for t in 0..n_periods {
        for i in 0..n_units {
            if d_arr[[t, i]] == 1.0 {
                first_treat_period = first_treat_period.min(t);
                break;
            }
        }
    }
    let treated_periods = n_periods.saturating_sub(first_treat_period);

    // Get control observations for LOOCV
    let control_obs = get_control_observations(&y_arr, &control_mask_arr);
    let n_attempted = control_obs.len();

    // Build grid combinations
    let mut grid_combinations: Vec<(f64, f64, f64)> = Vec::new();
    for &lt in &lambda_time_vec {
        for &lu in &lambda_unit_vec {
            for &ln in &lambda_nn_vec {
                grid_combinations.push((lt, lu, ln));
            }
        }
    }

    // Parallel grid search - try all combinations
    let results: Vec<(f64, f64, f64, f64, usize, Option<(usize, usize)>)> = grid_combinations
        .into_par_iter()
        .map(|(lt, lu, ln)| {
            // Convert λ_nn=∞ → 1e10 (factor model disabled)
            let ln_eff = if ln.is_infinite() { 1e10 } else { ln };

            let (score, n_valid, first_failed) = loocv_score_joint(
                &y_arr,
                &d_arr,
                &control_obs,
                lt,
                lu,
                ln_eff,
                treated_periods,
                max_iter,
                tol,
            );

            (lt, lu, ln, score, n_valid, first_failed)
        })
        .collect();

    // Find best result
    let mut best_result = (
        lambda_time_vec.first().copied().unwrap_or(0.0),
        lambda_unit_vec.first().copied().unwrap_or(0.0),
        lambda_nn_vec.first().copied().unwrap_or(0.0),
        f64::INFINITY,
        0usize,
        None,
    );

    for (lt, lu, ln, score, n_valid, first_failed) in results {
        if score < best_result.3 {
            best_result = (lt, lu, ln, score, n_valid, first_failed);
        }
    }

    let (best_lt, best_lu, best_ln, best_score, n_valid, first_failed) = best_result;

    Ok((
        best_lt,
        best_lu,
        best_ln,
        best_score,
        n_valid,
        n_attempted,
        first_failed,
    ))
}

/// Compute bootstrap variance estimation for TROP global method in parallel.
///
/// Performs unit-level block bootstrap, parallelizing across bootstrap iterations.
/// Uses stratified sampling to preserve treated/control unit ratio.
///
/// # Arguments
/// * `y` - Outcome matrix (n_periods x n_units)
/// * `d` - Treatment indicator matrix (n_periods x n_units)
/// * `lambda_time` - Selected time decay parameter
/// * `lambda_unit` - Selected unit distance parameter
/// * `lambda_nn` - Selected nuclear norm parameter
/// * `n_bootstrap` - Number of bootstrap iterations
/// * `max_iter` - Maximum iterations for model estimation
/// * `tol` - Convergence tolerance
/// * `control_indices` - Pre-generated stratified bootstrap indices for the
///   control pool, shape `(n_bootstrap, n_control_units)`, dtype `i64`.
///   Values must be in `[0, n_control_units)`.
/// * `treated_indices` - Pre-generated stratified bootstrap indices for the
///   treated pool, shape `(n_bootstrap, n_treated_units)`, dtype `i64`.
///   Values must be in `[0, n_treated_units)`.
/// * `survey_weights` - Optional unit-level survey weights (length n_units).
///   When provided, ATT is computed as a weighted mean of per-observation
///   treatment effects using unit weights. Model fitting, LOOCV, and distance
///   computation are unchanged.
///
/// The index arrays carry the RNG contract: they are produced on the Python
/// side by `diff_diff.bootstrap_utils.stratified_bootstrap_indices` with a
/// numpy `default_rng(seed)`, so Rust and Python consumers see identical
/// sampling under the same seed. Invalid index values (negative or out of
/// range) raise a `PyValueError` rather than silently producing malformed
/// bootstrap samples.
///
/// # Returns
/// (bootstrap_estimates, standard_error)
#[pyfunction]
#[pyo3(signature = (y, d, lambda_time, lambda_unit, lambda_nn, n_bootstrap, max_iter, tol, control_indices, treated_indices, survey_weights=None))]
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_trop_variance_global<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    d: PyReadonlyArray2<'py, f64>,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: usize,
    max_iter: usize,
    tol: f64,
    control_indices: PyReadonlyArray2<'py, i64>,
    treated_indices: PyReadonlyArray2<'py, i64>,
    survey_weights: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let y_arr = y.as_array().to_owned();
    let d_arr = d.as_array().to_owned();
    let ctrl_idx_arr = control_indices.as_array().to_owned();
    let trt_idx_arr = treated_indices.as_array().to_owned();
    let sw_arr: Option<Array1<f64>> = survey_weights.map(|sw| sw.as_array().to_owned());

    let n_units = y_arr.ncols();
    let n_periods = y_arr.nrows();

    // Identify treated and control units for stratified sampling
    let mut original_treated_units: Vec<usize> = Vec::new();
    let mut original_control_units: Vec<usize> = Vec::new();
    for i in 0..n_units {
        let is_ever_treated = (0..n_periods).any(|t| d_arr[[t, i]] == 1.0);
        if is_ever_treated {
            original_treated_units.push(i);
        } else {
            original_control_units.push(i);
        }
    }
    let n_treated_units = original_treated_units.len();
    let n_control_units = original_control_units.len();

    // Validate index-array shapes match the stratified pool sizes
    if ctrl_idx_arr.shape() != [n_bootstrap, n_control_units] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "control_indices shape {:?} does not match (n_bootstrap={}, n_control_units={})",
            ctrl_idx_arr.shape(),
            n_bootstrap,
            n_control_units,
        )));
    }
    if trt_idx_arr.shape() != [n_bootstrap, n_treated_units] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "treated_indices shape {:?} does not match (n_bootstrap={}, n_treated_units={})",
            trt_idx_arr.shape(),
            n_bootstrap,
            n_treated_units,
        )));
    }

    // Validate index values are in range. Fail fast with a clean PyValueError
    // rather than panicking inside the parallel loop on a negative cast or an
    // out-of-pool Vec index.
    if n_control_units > 0 {
        let n_ctrl = n_control_units as i64;
        for v in ctrl_idx_arr.iter() {
            if *v < 0 || *v >= n_ctrl {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "control_indices contains out-of-range value {} (valid: [0, {}))",
                    v, n_control_units,
                )));
            }
        }
    }
    if n_treated_units > 0 {
        let n_trt = n_treated_units as i64;
        for v in trt_idx_arr.iter() {
            if *v < 0 || *v >= n_trt {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "treated_indices contains out-of-range value {} (valid: [0, {}))",
                    v, n_treated_units,
                )));
            }
        }
    }

    // Determine treated periods from D matrix
    let mut first_treat_period = n_periods;
    for t in 0..n_periods {
        for i in 0..n_units {
            if d_arr[[t, i]] == 1.0 {
                first_treat_period = first_treat_period.min(t);
                break;
            }
        }
    }
    let treated_periods = n_periods.saturating_sub(first_treat_period);

    // Convert λ_nn=∞ → 1e10 (factor model disabled)
    let ln_eff = if lambda_nn.is_infinite() {
        1e10
    } else {
        lambda_nn
    };

    // Run bootstrap iterations in parallel
    // RNG-canonical contract: control_indices and treated_indices are pre-generated
    // by numpy.random.default_rng(seed) on the Python side via
    // diff_diff.bootstrap_utils.stratified_bootstrap_indices, so SE is identical
    // across backends under the same seed (silent-failures finding #23).
    let bootstrap_estimates: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .filter_map(|b| {
            // Stratified sampling: consume pre-generated indices for replicate b
            let mut sampled_units: Vec<usize> = Vec::with_capacity(n_units);
            for j in 0..n_control_units {
                let idx = ctrl_idx_arr[[b, j]] as usize;
                sampled_units.push(original_control_units[idx]);
            }
            for j in 0..n_treated_units {
                let idx = trt_idx_arr[[b, j]] as usize;
                sampled_units.push(original_treated_units[idx]);
            }

            // Create bootstrap matrices by selecting columns
            let mut y_boot = Array2::<f64>::zeros((n_periods, n_units));
            let mut d_boot = Array2::<f64>::zeros((n_periods, n_units));

            for (new_idx, &old_idx) in sampled_units.iter().enumerate() {
                for t in 0..n_periods {
                    y_boot[[t, new_idx]] = y_arr[[t, old_idx]];
                    d_boot[[t, new_idx]] = d_arr[[t, old_idx]];
                }
            }

            // Compute weights and fit global model
            let delta = compute_joint_weights(
                &y_boot.view(),
                &d_boot.view(),
                lambda_time,
                lambda_unit,
                treated_periods,
            );

            let result = if ln_eff >= 1e10 {
                solve_joint_no_lowrank(&y_boot.view(), &delta.view()).map(|(mu, alpha, beta)| {
                    let l = Array2::<f64>::zeros((n_periods, n_units));
                    (mu, alpha, beta, l)
                })
            } else {
                solve_joint_with_lowrank(&y_boot.view(), &delta.view(), ln_eff, max_iter, tol)
            };

            // Post-hoc tau extraction: ATT = mean(Y - mu - alpha - beta - L) over treated
            // When survey weights are provided, ATT is a weighted mean using unit-level weights.
            result.and_then(|(mu, alpha, beta, l)| {
                let mut tau_sum = 0.0;
                let mut weight_sum = 0.0;
                let mut tau_count = 0;
                for t in 0..n_periods {
                    for i in 0..n_units {
                        if d_boot[[t, i]] == 1.0 && y_boot[[t, i]].is_finite() {
                            let tau = y_boot[[t, i]] - mu - alpha[i] - beta[t] - l[[t, i]];
                            let w = match &sw_arr {
                                Some(sw) => sw[sampled_units[i]],
                                None => 1.0,
                            };
                            tau_sum += w * tau;
                            weight_sum += w;
                            tau_count += 1;
                        }
                    }
                }
                if tau_count > 0 {
                    Some(tau_sum / weight_sum)
                } else {
                    None
                }
            })
        })
        .collect();

    // Compute standard error
    // Return NaN when < 2 samples to properly propagate undefined inference
    let se = if bootstrap_estimates.len() < 2 {
        f64::NAN
    } else {
        let n = bootstrap_estimates.len() as f64;
        let mean = bootstrap_estimates.iter().sum::<f64>() / n;
        let variance = bootstrap_estimates
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        variance.sqrt()
    };

    let estimates_arr = Array1::from_vec(bootstrap_estimates);
    Ok((estimates_arr.to_pyarray(py), se))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_compute_pair_distance() {
        let y_j = array![1.0, 2.0, 3.0, 4.0];
        let y_i = array![1.5, 2.5, 3.5, 4.5];
        let valid_j = array![true, true, true, true];
        let valid_i = array![true, true, true, true];

        let dist =
            compute_pair_distance(&y_j.view(), &y_i.view(), &valid_j.view(), &valid_i.view());

        // RMSE of constant difference 0.5 should be 0.5
        assert!((dist - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_pair_distance_partial_overlap() {
        let y_j = array![1.0, 2.0, 3.0, 4.0];
        let y_i = array![1.5, 2.5, 3.5, 4.5];
        let valid_j = array![true, true, false, false];
        let valid_i = array![true, false, true, false];

        // Only period 0 overlaps
        let dist =
            compute_pair_distance(&y_j.view(), &y_i.view(), &valid_j.view(), &valid_i.view());

        // RMSE of single difference 0.5 should be 0.5
        assert!((dist - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_pair_distance_no_overlap() {
        let y_j = array![1.0, 2.0, 3.0, 4.0];
        let y_i = array![1.5, 2.5, 3.5, 4.5];
        let valid_j = array![true, true, false, false];
        let valid_i = array![false, false, true, true];

        let dist =
            compute_pair_distance(&y_j.view(), &y_i.view(), &valid_j.view(), &valid_i.view());

        assert!(dist.is_infinite());
    }

    #[test]
    fn test_unit_distance_matrix_diagonal_zero() {
        let y = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let d = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];

        let dist = compute_unit_distance_matrix_internal(&y.view(), &d.view());

        // Diagonal should be 0
        for i in 0..3 {
            assert!((dist[[i, i]]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_unit_distance_matrix_symmetric() {
        let y = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let d = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];

        let dist = compute_unit_distance_matrix_internal(&y.view(), &d.view());

        // Matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((dist[[i, j]] - dist[[j, i]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_max_abs_diff() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.1, 1.9, 3.5];

        let diff = max_abs_diff(&a, &b);
        assert!((diff - 0.5).abs() < 1e-10);
    }
}
