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
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

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
) -> PyResult<&'py PyArray2<f64>> {
    let y_arr = y.as_array();
    let d_arr = d.as_array();

    let dist_matrix = compute_unit_distance_matrix_internal(&y_arr, &d_arr);

    Ok(dist_matrix.into_pyarray(py))
}

/// Internal implementation of unit distance matrix computation.
///
/// Parallelizes over unit pairs using rayon.
fn compute_unit_distance_matrix_internal(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
) -> Array2<f64> {
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
/// * `fixed_time` - Fixed lambda_time (inf for disabled)
/// * `fixed_unit` - Fixed lambda_unit (inf for disabled)
/// * `fixed_nn` - Fixed lambda_nn (inf for disabled)
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
            // Set parameters, converting inf for "disabled" mode
            // Per paper Equations 2-3:
            // - λ_time/λ_unit=∞ → uniform weights → use 0.0
            // - λ_nn=∞ → infinite penalty → L≈0 (factor model disabled) → use 1e10
            // Note: λ_nn=0 means NO regularization (full-rank L), opposite of "disabled"
            //
            // IMPORTANT: Convert the grid value BEFORE using it, matching Python behavior.
            // This ensures Rust and Python evaluate the same objective for infinity grids.
            let (lambda_time, lambda_unit, lambda_nn) = match param_type {
                0 => {
                    // Searching λ_time: convert grid value if infinite
                    let value_converted = if value.is_infinite() { 0.0 } else { value };
                    (value_converted,
                     if fixed_unit.is_infinite() { 0.0 } else { fixed_unit },
                     if fixed_nn.is_infinite() { 1e10 } else { fixed_nn })
                },
                1 => {
                    // Searching λ_unit: convert grid value if infinite
                    let value_converted = if value.is_infinite() { 0.0 } else { value };
                    (if fixed_time.is_infinite() { 0.0 } else { fixed_time },
                     value_converted,
                     if fixed_nn.is_infinite() { 1e10 } else { fixed_nn })
                },
                _ => {
                    // Searching λ_nn: convert grid value if infinite
                    let value_converted = if value.is_infinite() { 1e10 } else { value };
                    (if fixed_time.is_infinite() { 0.0 } else { fixed_time },
                     if fixed_unit.is_infinite() { 0.0 } else { fixed_unit },
                     value_converted)
                },
            };

            let (score, _, _) = loocv_score_for_params(
                y, d, control_mask, time_dist, control_obs,
                lambda_time, lambda_unit, lambda_nn,
                max_iter, tol,
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
            y, d, control_mask, time_dist, control_obs,
            lambda_unit_grid, lambda_time, 0.0, lambda_nn, 1, max_iter, tol,
        );
        lambda_unit = new_unit;

        // Optimize λ_time (fix λ_unit, λ_nn)
        let (new_time, _) = univariate_loocv_search(
            y, d, control_mask, time_dist, control_obs,
            lambda_time_grid, 0.0, lambda_unit, lambda_nn, 0, max_iter, tol,
        );
        lambda_time = new_time;

        // Optimize λ_nn (fix λ_unit, λ_time)
        let (new_nn, score) = univariate_loocv_search(
            y, d, control_mask, time_dist, control_obs,
            lambda_nn_grid, lambda_time, lambda_unit, 0.0, 2, max_iter, tol,
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
/// * `max_loocv_samples` - Maximum control observations to evaluate
/// * `max_iter` - Maximum iterations for model estimation
/// * `tol` - Convergence tolerance
/// * `seed` - Random seed for subsampling
///
/// # Returns
/// (best_lambda_time, best_lambda_unit, best_lambda_nn, best_score, n_valid, n_attempted, first_failed_obs)
/// where n_valid and n_attempted are the counts for the best parameter combination,
/// allowing Python to emit warnings when >10% of fits fail.
/// first_failed_obs is Some((t, i)) if a fit failed during final score computation, None otherwise.
#[pyfunction]
#[pyo3(signature = (y, d, control_mask, time_dist_matrix, lambda_time_grid, lambda_unit_grid, lambda_nn_grid, max_loocv_samples, max_iter, tol, seed))]
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
    max_loocv_samples: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> PyResult<(f64, f64, f64, f64, usize, usize, Option<(usize, usize)>)> {
    let y_arr = y.as_array();
    let d_arr = d.as_array();
    let control_mask_arr = control_mask.as_array();
    let time_dist_arr = time_dist_matrix.as_array();
    let lambda_time_vec: Vec<f64> = lambda_time_grid.as_array().to_vec();
    let lambda_unit_vec: Vec<f64> = lambda_unit_grid.as_array().to_vec();
    let lambda_nn_vec: Vec<f64> = lambda_nn_grid.as_array().to_vec();

    // Get control observations for LOOCV
    let control_obs = get_control_observations(
        &y_arr,
        &control_mask_arr,
        max_loocv_samples,
        seed,
    );

    let n_attempted = control_obs.len();

    // Stage 1: Univariate searches for initial values (paper footnote 2)
    // λ_time search: fix λ_unit=0, λ_nn=∞ (disabled)
    let (lambda_time_init, _) = univariate_loocv_search(
        &y_arr, &d_arr, &control_mask_arr, &time_dist_arr, &control_obs,
        &lambda_time_vec, 0.0, 0.0, f64::INFINITY, 0, max_iter, tol,
    );

    // λ_nn search: fix λ_time=∞ (disabled), λ_unit=0
    let (lambda_nn_init, _) = univariate_loocv_search(
        &y_arr, &d_arr, &control_mask_arr, &time_dist_arr, &control_obs,
        &lambda_nn_vec, f64::INFINITY, 0.0, 0.0, 2, max_iter, tol,
    );

    // λ_unit search: fix λ_nn=∞, λ_time=0
    let (lambda_unit_init, _) = univariate_loocv_search(
        &y_arr, &d_arr, &control_mask_arr, &time_dist_arr, &control_obs,
        &lambda_unit_vec, 0.0, 0.0, f64::INFINITY, 1, max_iter, tol,
    );

    // Stage 2: Cycling refinement
    let (best_time, best_unit, best_nn) = cycling_parameter_search(
        &y_arr, &d_arr, &control_mask_arr, &time_dist_arr, &control_obs,
        &lambda_time_vec, &lambda_unit_vec, &lambda_nn_vec,
        lambda_time_init, lambda_unit_init, lambda_nn_init,
        max_iter, tol, 10,
    );

    // Convert infinity values BEFORE computing final score (Issue 1 fix)
    // Per paper Equations 2-3:
    // - λ_time/λ_unit=∞ → uniform weights → use 0.0
    // - λ_nn=∞ → infinite penalty → L≈0 (factor model disabled) → use 1e10
    // This ensures final score computation matches what LOOCV evaluated.
    let best_time_eff = if best_time.is_infinite() { 0.0 } else { best_time };
    let best_unit_eff = if best_unit.is_infinite() { 0.0 } else { best_unit };
    let best_nn_eff = if best_nn.is_infinite() { 1e10 } else { best_nn };

    // Compute final score with converted values
    let (best_score, n_valid, first_failed) = loocv_score_for_params(
        &y_arr, &d_arr, &control_mask_arr, &time_dist_arr, &control_obs,
        best_time_eff, best_unit_eff, best_nn_eff,
        max_iter, tol,
    );

    // Return ORIGINAL grid values (for user visibility) but score computed with converted
    Ok((best_time, best_unit, best_nn, best_score, n_valid, n_attempted, first_failed))
}

/// Get sampled control observations for LOOCV.
fn get_control_observations(
    y: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    max_samples: usize,
    seed: u64,
) -> Vec<(usize, usize)> {
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;

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

    // Subsample if needed
    if obs.len() > max_samples {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        obs.shuffle(&mut rng);
        obs.truncate(max_samples);
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

    let mut tau_sq_sum = 0.0;
    let mut n_valid = 0usize;

    for &(t, i) in control_obs {
        // Compute observation-specific weight matrix
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

        // Estimate model excluding this observation
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
                // Pseudo treatment effect: τ = Y - α - β - L
                let tau = y[[t, i]] - alpha[i] - beta[t] - l[[t, i]];
                tau_sq_sum += tau * tau;
                n_valid += 1;
            }
            None => {
                // Per Equation 5: Q(λ) must sum over ALL D==0 cells
                // Any failure means this λ cannot produce valid estimates for all cells
                // Return the failed observation (t, i) for warning metadata
                return (f64::INFINITY, n_valid, Some((t, i)));
            }
        }
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
        if d[[t, i]] == 0.0 && d[[t, j]] == 0.0
            && y[[t, i]].is_finite() && y[[t, j]].is_finite()
        {
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
/// - ALL units get weights (not just those untreated at target period)
/// - The (1 - D_js) masking in the loss naturally excludes treated cells
/// - Weights are normalized to sum to 1 (probability weights)
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
    let mut time_weights: Array1<f64> = Array1::from_shape_fn(n_periods, |s| {
        let dist = time_dist[[target_period, s]] as f64;
        (-lambda_time * dist).exp()
    });

    // Normalize time weights to sum to 1
    let time_sum: f64 = time_weights.sum();
    if time_sum > 0.0 {
        time_weights /= time_sum;
    }

    // Unit weights: ω_j = exp(-λ_unit × dist(j, i))
    // Paper alignment: compute for ALL units, let control masking handle exclusion
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

    // Normalize unit weights to sum to 1
    let unit_sum: f64 = unit_weights.sum();
    if unit_sum > 0.0 {
        unit_weights /= unit_sum;
    }

    // Outer product: W[t, i] = time_weights[t] * unit_weights[i]
    // Result is normalized since both components sum to 1
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
/// Paper alignment: Uses weighted proximal gradient for L update:
///   L ← prox_{η·λ_nn·||·||_*}(L + η·(W ⊙ (R - L)))
/// where η ≤ 1/max(W) for convergence.
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
    let mut est_mask = Array2::<bool>::from_shape_fn((n_periods, n_units), |(t, i)| {
        control_mask[[t, i]] != 0
    });

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

    // Compute step size for proximal gradient: η ≤ 1/max(W)
    let w_max = w_masked.iter().cloned().fold(0.0_f64, f64::max);
    let eta = if w_max > 0.0 { 1.0 / w_max } else { 1.0 };

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
        // Paper alignment: Use proximal gradient instead of direct soft-thresholding
        // L ← prox_{η·λ_nn·||·||_*}(L + η·(W ⊙ (R - L)))
        // where R = Y - α - β

        // Compute target residual R = Y - α - β
        let mut r_target = Array2::<f64>::zeros((n_periods, n_units));
        for t in 0..n_periods {
            for i in 0..n_units {
                r_target[[t, i]] = y_safe[[t, i]] - alpha[i] - beta[t];
            }
        }

        // Weighted proximal gradient step:
        // gradient_step = L + η * W ⊙ (R - L)
        // For W=0 cells (treated obs), this keeps L unchanged
        let mut gradient_step = Array2::<f64>::zeros((n_periods, n_units));
        for t in 0..n_periods {
            for i in 0..n_units {
                gradient_step[[t, i]] = l[[t, i]] + eta * w_masked[[t, i]] * (r_target[[t, i]] - l[[t, i]]);
            }
        }

        // Proximal step: soft-threshold singular values with scaled lambda
        l = soft_threshold_svd(&gradient_step, eta * lambda_nn)?;

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

    // Compute SVD using ndarray-linalg
    use ndarray_linalg::SVD;

    let (u, s, vt) = match m.svd(true, true) {
        Ok((Some(u), s, Some(vt))) => (u, s, vt),
        _ => return Some(Array2::zeros(m.raw_dim())),
    };

    // Check for non-finite SVD output
    if !u.iter().all(|&x| x.is_finite())
        || !s.iter().all(|&x| x.is_finite())
        || !vt.iter().all(|&x| x.is_finite())
    {
        return Some(Array2::zeros(m.raw_dim()));
    }

    // Soft-threshold singular values
    let s_thresh: Array1<f64> = s.mapv(|sv| (sv - threshold).max(0.0));

    // Count non-zero singular values
    let nonzero_count = s_thresh.iter().filter(|&&sv| sv > 1e-10).count();

    if nonzero_count == 0 {
        return Some(Array2::zeros(m.raw_dim()));
    }

    // Truncated reconstruction: U @ diag(s_thresh) @ Vt
    let n_rows = m.nrows();
    let n_cols = m.ncols();
    let mut result = Array2::<f64>::zeros((n_rows, n_cols));

    for k in 0..nonzero_count {
        if s_thresh[k] > 1e-10 {
            for i in 0..n_rows {
                for j in 0..n_cols {
                    result[[i, j]] += s_thresh[k] * u[[i, k]] * vt[[k, j]];
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

/// Compute bootstrap variance estimation for TROP in parallel.
///
/// Performs unit-level block bootstrap, parallelizing across bootstrap iterations.
///
/// # Arguments
/// * `y` - Outcome matrix (n_periods x n_units)
/// * `d` - Treatment indicator matrix (n_periods x n_units)
/// * `control_mask` - Boolean mask for control observations
/// * `control_unit_idx` - Array of control unit indices
/// * `treated_obs` - List of (t, i) treated observations
/// * `unit_dist_matrix` - Pre-computed unit distance matrix
/// * `time_dist_matrix` - Pre-computed time distance matrix
/// * `lambda_time` - Selected time decay parameter
/// * `lambda_unit` - Selected unit distance parameter
/// * `lambda_nn` - Selected nuclear norm parameter
/// * `n_bootstrap` - Number of bootstrap iterations
/// * `max_iter` - Maximum iterations for model estimation
/// * `tol` - Convergence tolerance
/// * `seed` - Random seed
///
/// # Returns
/// (bootstrap_estimates, standard_error)
#[pyfunction]
#[pyo3(signature = (y, d, control_mask, time_dist_matrix, lambda_time, lambda_unit, lambda_nn, n_bootstrap, max_iter, tol, seed))]
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
    seed: u64,
) -> PyResult<(&'py PyArray1<f64>, f64)> {
    let y_arr = y.as_array().to_owned();
    let d_arr = d.as_array().to_owned();
    let control_mask_arr = control_mask.as_array().to_owned();
    let time_dist_arr = time_dist_matrix.as_array().to_owned();

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

    // Run bootstrap iterations in parallel
    let bootstrap_estimates: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .filter_map(|b| {
            use rand::prelude::*;
            use rand_xoshiro::Xoshiro256PlusPlus;

            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(b as u64));

            // Issue D fix: Stratified sampling - sample control and treated units separately
            let mut sampled_units: Vec<usize> = Vec::with_capacity(n_units);

            // Sample control units with replacement
            for _ in 0..n_control_units {
                let idx = rng.gen_range(0..n_control_units);
                sampled_units.push(original_control_units[idx]);
            }

            // Sample treated units with replacement
            for _ in 0..n_treated_units {
                let idx = rng.gen_range(0..n_treated_units);
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
            let mut tau_values = Vec::with_capacity(boot_treated.len());

            for (t, i) in boot_treated {
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
                    tau_values.push(tau);
                }
            }

            if tau_values.is_empty() {
                None
            } else {
                Some(tau_values.iter().sum::<f64>() / tau_values.len() as f64)
            }
        })
        .collect();

    // Compute standard error
    let se = if bootstrap_estimates.len() < 2 {
        0.0
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
    Ok((estimates_arr.into_pyarray(py), se))
}

// ============================================================================
// Joint method implementation
// ============================================================================

/// Compute global weights for joint method estimation.
///
/// Unlike twostep (which computes per-observation weights), joint uses global
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
    let mut average_treated = Array1::<f64>::zeros(n_periods);
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
                0.0
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

    delta
}

/// Solve joint TWFE + treatment via weighted least squares (no low-rank).
///
/// Minimizes: min Σ δ_{it}(Y_{it} - μ - α_i - β_t - τ*W_{it})²
///
/// # Returns
/// (mu, alpha, beta, tau) estimated parameters
fn solve_joint_no_lowrank(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    delta: &ArrayView2<f64>,
) -> Option<(f64, Array1<f64>, Array1<f64>, f64)> {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    // We solve using normal equations with the design matrix structure
    // Rather than build full X matrix, use block structure for efficiency
    //
    // The model: Y_it = μ + α_i + β_t + τ*D_it + ε_it
    // With identification: α_0 = β_0 = 0

    // Compute weighted sums needed for normal equations
    let mut sum_w = 0.0;
    let mut sum_wy = 0.0;

    // Per-unit and per-period weighted sums
    let mut sum_w_by_unit = Array1::<f64>::zeros(n_units);
    let mut sum_wy_by_unit = Array1::<f64>::zeros(n_units);
    let mut sum_w_by_period = Array1::<f64>::zeros(n_periods);
    let mut sum_wy_by_period = Array1::<f64>::zeros(n_periods);

    for t in 0..n_periods {
        for i in 0..n_units {
            let w = delta[[t, i]];
            let y_ti = if y[[t, i]].is_finite() { y[[t, i]] } else { 0.0 };

            sum_w += w;
            sum_wy += w * y_ti;

            sum_w_by_unit[i] += w;
            sum_wy_by_unit[i] += w * y_ti;
            sum_w_by_period[t] += w;
            sum_wy_by_period[t] += w * y_ti;
        }
    }

    if sum_w < 1e-10 {
        return None;
    }

    // Use iterative approach: alternate between (alpha, beta, tau) and mu
    // until convergence (simpler than full normal equations)
    let mut mu = sum_wy / sum_w;
    let mut alpha = Array1::<f64>::zeros(n_units);
    let mut beta = Array1::<f64>::zeros(n_periods);
    let mut tau = 0.0;

    for _ in 0..50 {
        let mu_old = mu;
        let tau_old = tau;

        // Update alpha (fixing beta, tau, mu)
        for i in 1..n_units {  // α_0 = 0 for identification
            if sum_w_by_unit[i] > 1e-10 {
                let mut num = 0.0;
                for t in 0..n_periods {
                    let w = delta[[t, i]];
                    let y_ti = if y[[t, i]].is_finite() { y[[t, i]] } else { 0.0 };
                    num += w * (y_ti - mu - beta[t] - tau * d[[t, i]]);
                }
                alpha[i] = num / sum_w_by_unit[i];
            }
        }

        // Update beta (fixing alpha, tau, mu)
        for t in 1..n_periods {  // β_0 = 0 for identification
            if sum_w_by_period[t] > 1e-10 {
                let mut num = 0.0;
                for i in 0..n_units {
                    let w = delta[[t, i]];
                    let y_ti = if y[[t, i]].is_finite() { y[[t, i]] } else { 0.0 };
                    num += w * (y_ti - mu - alpha[i] - tau * d[[t, i]]);
                }
                beta[t] = num / sum_w_by_period[t];
            }
        }

        // Update tau (fixing alpha, beta, mu)
        let mut num_tau = 0.0;
        let mut denom_tau = 0.0;
        for t in 0..n_periods {
            for i in 0..n_units {
                let w = delta[[t, i]];
                let y_ti = if y[[t, i]].is_finite() { y[[t, i]] } else { 0.0 };
                let d_ti = d[[t, i]];
                if d_ti > 0.5 {  // Only treated observations contribute
                    num_tau += w * d_ti * (y_ti - mu - alpha[i] - beta[t]);
                    denom_tau += w * d_ti * d_ti;
                }
            }
        }
        if denom_tau > 1e-10 {
            tau = num_tau / denom_tau;
        }

        // Update mu (fixing alpha, beta, tau)
        let mut num_mu = 0.0;
        for t in 0..n_periods {
            for i in 0..n_units {
                let w = delta[[t, i]];
                let y_ti = if y[[t, i]].is_finite() { y[[t, i]] } else { 0.0 };
                num_mu += w * (y_ti - alpha[i] - beta[t] - tau * d[[t, i]]);
            }
        }
        mu = num_mu / sum_w;

        // Check convergence
        if (mu - mu_old).abs() < 1e-8 && (tau - tau_old).abs() < 1e-8 {
            break;
        }
    }

    Some((mu, alpha, beta, tau))
}

/// Solve joint TWFE + treatment + low-rank via alternating minimization.
///
/// Minimizes: min Σ δ_{it}(Y_{it} - μ - α_i - β_t - L_{it} - τ*W_{it})² + λ_nn||L||_*
///
/// # Returns
/// (mu, alpha, beta, L, tau) estimated parameters
fn solve_joint_with_lowrank(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    delta: &ArrayView2<f64>,
    lambda_nn: f64,
    max_iter: usize,
    tol: f64,
) -> Option<(f64, Array1<f64>, Array1<f64>, Array2<f64>, f64)> {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    // Initialize L = 0
    let mut l = Array2::<f64>::zeros((n_periods, n_units));

    for _ in 0..max_iter {
        let l_old = l.clone();

        // Step 1: Fix L, solve for (mu, alpha, beta, tau)
        // Adjusted outcome: Y - L
        let y_adj = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
            let y_ti = if y[[t, i]].is_finite() { y[[t, i]] } else { 0.0 };
            y_ti - l[[t, i]]
        });

        let (mu, alpha, beta, tau) = solve_joint_no_lowrank(&y_adj.view(), d, delta)?;

        // Step 2: Fix (mu, alpha, beta, tau), update L
        // Residual: R = Y - mu - alpha - beta - tau*D
        let mut r = Array2::<f64>::zeros((n_periods, n_units));
        for t in 0..n_periods {
            for i in 0..n_units {
                let y_ti = if y[[t, i]].is_finite() { y[[t, i]] } else { 0.0 };
                r[[t, i]] = y_ti - mu - alpha[i] - beta[t] - tau * d[[t, i]];
            }
        }

        // Weighted proximal step for L (soft-threshold SVD)
        let delta_max = delta.iter().cloned().fold(0.0_f64, f64::max);
        let eta = if delta_max > 0.0 { 1.0 / delta_max } else { 1.0 };

        // gradient_step = L + eta * delta * (R - L)
        let mut gradient_step = Array2::<f64>::zeros((n_periods, n_units));
        for t in 0..n_periods {
            for i in 0..n_units {
                let delta_norm = if delta_max > 0.0 {
                    delta[[t, i]] / delta_max
                } else {
                    delta[[t, i]]
                };
                gradient_step[[t, i]] = l[[t, i]] + delta_norm * (r[[t, i]] - l[[t, i]]);
            }
        }

        // Soft-threshold singular values
        l = soft_threshold_svd(&gradient_step, eta * lambda_nn)?;

        // Check convergence
        let l_diff = max_abs_diff_2d(&l, &l_old);
        if l_diff < tol {
            break;
        }
    }

    // Final solve with converged L
    let y_adj = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
        let y_ti = if y[[t, i]].is_finite() { y[[t, i]] } else { 0.0 };
        y_ti - l[[t, i]]
    });
    let (mu, alpha, beta, tau) = solve_joint_no_lowrank(&y_adj.view(), d, delta)?;

    Some((mu, alpha, beta, l, tau))
}

/// Compute LOOCV score for joint method with specific parameter combination.
///
/// Following paper's Equation 5:
/// Q(λ) = Σ_{j,s: D_js=0} [τ̂_js^loocv(λ)]²
///
/// For joint method, we exclude each control observation, fit the joint model
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

    let mut tau_sq_sum = 0.0;
    let mut n_valid = 0usize;

    // Compute global weights (same for all LOOCV iterations)
    let delta = compute_joint_weights(y, d, lambda_time, lambda_unit, treated_periods);

    for &(t_ex, i_ex) in control_obs {
        // Create modified delta with excluded observation zeroed out
        let mut delta_ex = delta.clone();
        delta_ex[[t_ex, i_ex]] = 0.0;

        // Fit joint model excluding this observation
        let result = if lambda_nn >= 1e10 {
            solve_joint_no_lowrank(y, d, &delta_ex.view())
                .map(|(mu, alpha, beta, tau)| {
                    let l = Array2::<f64>::zeros((n_periods, n_units));
                    (mu, alpha, beta, l, tau)
                })
        } else {
            solve_joint_with_lowrank(y, d, &delta_ex.view(), lambda_nn, max_iter, tol)
        };

        match result {
            Some((mu, alpha, beta, l, _tau)) => {
                // Pseudo treatment effect: τ = Y - μ - α - β - L
                let y_ti = if y[[t_ex, i_ex]].is_finite() {
                    y[[t_ex, i_ex]]
                } else {
                    continue;
                };
                let tau_loocv = y_ti - mu - alpha[i_ex] - beta[t_ex] - l[[t_ex, i_ex]];
                tau_sq_sum += tau_loocv * tau_loocv;
                n_valid += 1;
            }
            None => {
                // Any failure means this λ combination is invalid per Equation 5
                return (f64::INFINITY, n_valid, Some((t_ex, i_ex)));
            }
        }
    }

    if n_valid == 0 {
        (f64::INFINITY, 0, None)
    } else {
        (tau_sq_sum, n_valid, None)
    }
}

/// Perform LOOCV grid search for joint method using two-stage approach.
///
/// Following paper's footnote 2:
/// - Stage 1: Univariate searches for initial values with extreme fixed parameters
/// - Stage 2: Cycling (coordinate descent) until convergence
///
/// # Arguments
/// * `y` - Outcome matrix (n_periods x n_units)
/// * `d` - Treatment indicator matrix (n_periods x n_units)
/// * `control_mask` - Boolean mask (n_periods x n_units) for control observations
/// * `lambda_time_grid` - Grid of time decay parameters
/// * `lambda_unit_grid` - Grid of unit distance parameters
/// * `lambda_nn_grid` - Grid of nuclear norm parameters
/// * `max_loocv_samples` - Maximum control observations to evaluate
/// * `max_iter` - Maximum iterations for model estimation
/// * `tol` - Convergence tolerance
/// * `seed` - Random seed for subsampling
///
/// # Returns
/// (best_lambda_time, best_lambda_unit, best_lambda_nn, best_score, n_valid, n_attempted, first_failed_obs)
#[pyfunction]
#[pyo3(signature = (y, d, control_mask, lambda_time_grid, lambda_unit_grid, lambda_nn_grid, max_loocv_samples, max_iter, tol, seed))]
#[allow(clippy::too_many_arguments)]
pub fn loocv_grid_search_joint<'py>(
    _py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    d: PyReadonlyArray2<'py, f64>,
    control_mask: PyReadonlyArray2<'py, u8>,
    lambda_time_grid: PyReadonlyArray1<'py, f64>,
    lambda_unit_grid: PyReadonlyArray1<'py, f64>,
    lambda_nn_grid: PyReadonlyArray1<'py, f64>,
    max_loocv_samples: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> PyResult<(f64, f64, f64, f64, usize, usize, Option<(usize, usize)>)> {
    let y_arr = y.as_array();
    let d_arr = d.as_array();
    let control_mask_arr = control_mask.as_array();
    let lambda_time_vec: Vec<f64> = lambda_time_grid.as_array().to_vec();
    let lambda_unit_vec: Vec<f64> = lambda_unit_grid.as_array().to_vec();
    let lambda_nn_vec: Vec<f64> = lambda_nn_grid.as_array().to_vec();

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
    let control_obs = get_control_observations(&y_arr, &control_mask_arr, max_loocv_samples, seed);
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
            // Convert infinity values
            let lt_eff = if lt.is_infinite() { 0.0 } else { lt };
            let lu_eff = if lu.is_infinite() { 0.0 } else { lu };
            let ln_eff = if ln.is_infinite() { 1e10 } else { ln };

            let (score, n_valid, first_failed) = loocv_score_joint(
                &y_arr,
                &d_arr,
                &control_obs,
                lt_eff,
                lu_eff,
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

    Ok((best_lt, best_lu, best_ln, best_score, n_valid, n_attempted, first_failed))
}

/// Compute bootstrap variance estimation for TROP joint method in parallel.
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
/// * `seed` - Random seed
///
/// # Returns
/// (bootstrap_estimates, standard_error)
#[pyfunction]
#[pyo3(signature = (y, d, lambda_time, lambda_unit, lambda_nn, n_bootstrap, max_iter, tol, seed))]
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_trop_variance_joint<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    d: PyReadonlyArray2<'py, f64>,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> PyResult<(&'py PyArray1<f64>, f64)> {
    let y_arr = y.as_array().to_owned();
    let d_arr = d.as_array().to_owned();

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

    // Convert infinity values for computation
    let lt_eff = if lambda_time.is_infinite() { 0.0 } else { lambda_time };
    let lu_eff = if lambda_unit.is_infinite() { 0.0 } else { lambda_unit };
    let ln_eff = if lambda_nn.is_infinite() { 1e10 } else { lambda_nn };

    // Run bootstrap iterations in parallel
    let bootstrap_estimates: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .filter_map(|b| {
            use rand::prelude::*;
            use rand_xoshiro::Xoshiro256PlusPlus;

            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(b as u64));

            // Stratified sampling - sample control and treated units separately
            let mut sampled_units: Vec<usize> = Vec::with_capacity(n_units);

            // Sample control units with replacement
            for _ in 0..n_control_units {
                if n_control_units > 0 {
                    let idx = rng.gen_range(0..n_control_units);
                    sampled_units.push(original_control_units[idx]);
                }
            }

            // Sample treated units with replacement
            for _ in 0..n_treated_units {
                if n_treated_units > 0 {
                    let idx = rng.gen_range(0..n_treated_units);
                    sampled_units.push(original_treated_units[idx]);
                }
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

            // Compute weights and fit joint model
            let delta = compute_joint_weights(
                &y_boot.view(),
                &d_boot.view(),
                lt_eff,
                lu_eff,
                treated_periods,
            );

            let result = if ln_eff >= 1e10 {
                solve_joint_no_lowrank(&y_boot.view(), &d_boot.view(), &delta.view())
                    .map(|(_, _, _, tau)| tau)
            } else {
                solve_joint_with_lowrank(
                    &y_boot.view(),
                    &d_boot.view(),
                    &delta.view(),
                    ln_eff,
                    max_iter,
                    tol,
                )
                .map(|(_, _, _, _, tau)| tau)
            };

            result
        })
        .collect();

    // Compute standard error
    let se = if bootstrap_estimates.len() < 2 {
        0.0
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
    Ok((estimates_arr.into_pyarray(py), se))
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

        let dist = compute_pair_distance(&y_j.view(), &y_i.view(), &valid_j.view(), &valid_i.view());

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
        let dist = compute_pair_distance(&y_j.view(), &y_i.view(), &valid_j.view(), &valid_i.view());

        // RMSE of single difference 0.5 should be 0.5
        assert!((dist - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_pair_distance_no_overlap() {
        let y_j = array![1.0, 2.0, 3.0, 4.0];
        let y_i = array![1.5, 2.5, 3.5, 4.5];
        let valid_j = array![true, true, false, false];
        let valid_i = array![false, false, true, true];

        let dist = compute_pair_distance(&y_j.view(), &y_i.view(), &valid_j.view(), &valid_i.view());

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
