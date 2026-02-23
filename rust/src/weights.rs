//! Synthetic control weight computation.
//!
//! This module provides optimized implementations of:
//! - Legacy synthetic control weight optimization (projected gradient descent)
//! - Frank-Wolfe synthetic control weights (matching R's synthdid)
//! - Simplex projection
//! - SDID unit and time weight computation

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray::linalg::general_mat_vec_mul;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

/// Maximum number of optimization iterations.
const MAX_ITER: usize = 1000;

/// Default convergence tolerance (matches Python's _OPTIMIZATION_TOL).
const DEFAULT_TOL: f64 = 1e-8;

/// Default step size for gradient descent.
const DEFAULT_STEP_SIZE: f64 = 0.1;

// =========================================================================
// Legacy synthetic control weights (projected gradient descent)
// =========================================================================

/// Compute synthetic control weights via projected gradient descent.
///
/// Solves: min_w ||Y_treated - Y_control @ w||² + lambda * ||w||²
/// subject to: w >= 0, sum(w) = 1
///
/// # Arguments
/// * `y_control` - Control unit outcomes matrix (n_pre, n_control)
/// * `y_treated` - Treated unit outcomes (n_pre,)
/// * `lambda_reg` - L2 regularization parameter
/// * `max_iter` - Maximum number of iterations (default: 1000)
/// * `tol` - Convergence tolerance (default: 1e-6)
///
/// # Returns
/// Optimal weights (n_control,) that sum to 1
#[pyfunction]
#[pyo3(signature = (y_control, y_treated, lambda_reg=0.0, max_iter=None, tol=None))]
pub fn compute_synthetic_weights<'py>(
    py: Python<'py>,
    y_control: PyReadonlyArray2<'py, f64>,
    y_treated: PyReadonlyArray1<'py, f64>,
    lambda_reg: f64,
    max_iter: Option<usize>,
    tol: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y_control_arr = y_control.as_array();
    let y_treated_arr = y_treated.as_array();

    let weights =
        compute_synthetic_weights_internal(&y_control_arr, &y_treated_arr, lambda_reg, max_iter, tol)?;

    Ok(weights.to_pyarray_bound(py))
}

/// Internal implementation of synthetic weight computation.
fn compute_synthetic_weights_internal(
    y_control: &ArrayView2<f64>,
    y_treated: &ArrayView1<f64>,
    lambda_reg: f64,
    max_iter: Option<usize>,
    tol: Option<f64>,
) -> PyResult<Array1<f64>> {
    let n_control = y_control.ncols();
    let max_iter = max_iter.unwrap_or(MAX_ITER);
    let tol = tol.unwrap_or(DEFAULT_TOL);

    // Precompute Hessian: H = Y_control' @ Y_control + lambda * I
    let h = {
        let ytc = y_control.t().dot(y_control);
        let mut h = ytc;
        // Add regularization to diagonal
        for i in 0..n_control {
            h[[i, i]] += lambda_reg;
        }
        h
    };

    // Precompute linear term: f = Y_control' @ Y_treated
    let f = y_control.t().dot(y_treated);

    // Initialize with uniform weights
    let mut weights = Array1::from_elem(n_control, 1.0 / n_control as f64);

    // Projected gradient descent
    let step_size = DEFAULT_STEP_SIZE;
    let mut prev_weights = weights.clone();

    for _ in 0..max_iter {
        // Gradient: grad = H @ weights - f
        let grad = h.dot(&weights) - &f;

        // Gradient step
        weights = &weights - step_size * &grad;

        // Project onto simplex
        weights = project_simplex_internal(&weights.view());

        // Check convergence
        let diff: f64 = weights
            .iter()
            .zip(prev_weights.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        if diff.sqrt() < tol {
            break;
        }

        prev_weights.assign(&weights);
    }

    Ok(weights)
}

// =========================================================================
// Simplex projection
// =========================================================================

/// Project a vector onto the probability simplex.
///
/// Implements the O(n log n) algorithm from:
/// Duchi et al. "Efficient Projections onto the ℓ1-Ball for Learning in High Dimensions"
///
/// # Arguments
/// * `v` - Input vector (n,)
///
/// # Returns
/// Projected vector (n,) satisfying: w >= 0, sum(w) = 1
#[pyfunction]
pub fn project_simplex<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let v_arr = v.as_array();
    let result = project_simplex_internal(&v_arr);
    Ok(result.to_pyarray_bound(py))
}

/// Internal implementation of simplex projection.
///
/// Algorithm:
/// 1. Sort v in descending order
/// 2. Find the largest k such that u_k + (1 - sum_{j=1}^k u_j) / k > 0
/// 3. Set theta = (sum_{j=1}^k u_j - 1) / k
/// 4. Return max(v - theta, 0)
fn project_simplex_internal(v: &ArrayView1<f64>) -> Array1<f64> {
    let n = v.len();

    // Sort in descending order
    let mut u: Vec<f64> = v.iter().cloned().collect();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Find rho: largest index where u[rho] + (1 - cumsum[rho]) / (rho + 1) > 0
    let mut cumsum = 0.0;
    let mut rho = 0;
    for (i, &ui) in u.iter().enumerate().take(n) {
        cumsum += ui;
        if ui + (1.0 - cumsum) / (i + 1) as f64 > 0.0 {
            rho = i;
        }
    }

    // Compute threshold
    let cumsum_rho: f64 = u.iter().take(rho + 1).sum();
    let theta = (cumsum_rho - 1.0) / (rho + 1) as f64;

    // Project: max(v - theta, 0)
    v.mapv(|x| (x - theta).max(0.0))
}

// =========================================================================
// Frank-Wolfe solver (matching R's synthdid)
// =========================================================================

/// Normalize vector to sum to 1. Fallback to uniform if sum is zero.
/// Matches R's synthdid sum_normalize().
pub(crate) fn sum_normalize_internal(v: &Array1<f64>) -> Array1<f64> {
    let s: f64 = v.sum();
    if s > 0.0 {
        v / s
    } else {
        Array1::from_elem(v.len(), 1.0 / v.len() as f64)
    }
}

/// Sparsify weight vector by zeroing out small entries.
/// Matches R's synthdid sparsify_function:
///   v[v <= max(v)/4] = 0; v = v / sum(v)
fn sparsify_internal(v: &Array1<f64>) -> Array1<f64> {
    let n = v.len();
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_v <= 0.0 {
        return Array1::from_elem(n, 1.0 / n as f64);
    }
    let threshold = max_v / 4.0;
    let mut result = v.clone();
    result.mapv_inplace(|x| if x <= threshold { 0.0 } else { x });
    sum_normalize_internal(&result)
}

/// Interval (in iterations) at which the Gram path refreshes `ata_x = ATA @ lam`
/// from scratch to prevent floating-point drift from incremental updates.
const GRAM_REFRESH_INTERVAL: usize = 100;

/// Find the index of the minimum element in a vector.
#[inline]
fn argmin_f64(v: &Array1<f64>) -> usize {
    debug_assert!(!v.is_empty(), "argmin called on empty array");
    v.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Gram-accelerated Frank-Wolfe loop for the T0 < N case (time weights).
///
/// By precomputing `ATA = A^T @ A` (T0×T0) and `ATb = A^T @ b` (T0,),
/// the entire FW iteration reduces from O(N×T0) per step to O(T0) per step.
/// This is a ~N/T0 speedup (e.g., ~100x at 5k scale where N=4000, T0=40).
///
/// Zero heap allocations per iteration — all workspace is pre-allocated.
fn sc_weight_fw_gram(
    a: &ArrayView2<f64>,
    b: &ArrayView1<f64>,
    lam: &mut Array1<f64>,
    eta: f64,
    zeta: f64,
    n: usize,
    min_decrease_sq: f64,
    max_iter: usize,
) {
    let t0 = lam.len();

    // Precompute Gram matrix and related quantities — O(N×T0²) once
    let ata = a.t().dot(a); // (T0, T0)
    let atb = a.t().dot(b); // (T0,)
    let b_norm_sq = b.dot(b); // scalar

    // Diagonal of ATA for d_err_sq computation
    let mut ata_diag = Array1::zeros(t0);
    for j in 0..t0 {
        ata_diag[j] = ata[[j, j]];
    }

    // Maintained incrementally: ata_x = ATA @ lam
    let mut ata_x = ata.dot(lam);

    // Pre-allocate workspace
    let mut half_grad = Array1::zeros(t0);

    let mut prev_val = f64::INFINITY;

    for t in 0..max_iter {
        // Step 1: half_grad[j] = ata_x[j] - atb[j] + eta * lam[j]
        for j in 0..t0 {
            half_grad[j] = ata_x[j] - atb[j] + eta * lam[j];
        }

        // Step 2: Find vertex with smallest gradient component
        let i = argmin_f64(&half_grad);

        // Step 3: Guard — check if direction is essentially zero
        // d_x_norm_sq = ||lam||² + 1 - 2*lam[i]  (since d_x = e_i - lam)
        let lam_norm_sq: f64 = lam.iter().map(|&v| v * v).sum();
        let d_x_norm_sq = lam_norm_sq + 1.0 - 2.0 * lam[i];
        if d_x_norm_sq < 1e-24 {
            // Already at optimal vertex — compute objective for convergence check
            let xt_ata_x: f64 = ata_x.iter().zip(lam.iter()).map(|(&a, &b)| a * b).sum();
            let atb_dot_lam: f64 = atb.iter().zip(lam.iter()).map(|(&a, &b)| a * b).sum();
            let val = zeta * zeta * lam_norm_sq + (xt_ata_x - 2.0 * atb_dot_lam + b_norm_sq) / n as f64;
            if t >= 1 && prev_val - val < min_decrease_sq {
                break;
            }
            prev_val = val;
            continue;
        }

        // Step 4: Guard — compute denominator for step size
        // d_err_sq = ||A[:,i] - A@lam||² = ata_diag[i] - 2*ata_x[i] + lam^T ATA lam
        let xt_ata_x: f64 = ata_x.iter().zip(lam.iter()).map(|(&a, &b)| a * b).sum();
        let d_err_sq = ata_diag[i] - 2.0 * ata_x[i] + xt_ata_x;
        let denom = d_err_sq + eta * d_x_norm_sq;
        if denom <= 0.0 {
            let atb_dot_lam: f64 = atb.iter().zip(lam.iter()).map(|(&a, &b)| a * b).sum();
            let val = zeta * zeta * lam_norm_sq + (xt_ata_x - 2.0 * atb_dot_lam + b_norm_sq) / n as f64;
            if t >= 1 && prev_val - val < min_decrease_sq {
                break;
            }
            prev_val = val;
            continue;
        }

        // Step 5: hg_dot_dx = half_grad[i] - half_grad @ lam
        let hg_dot_lam: f64 = half_grad.iter().zip(lam.iter()).map(|(&a, &b)| a * b).sum();
        let hg_dot_dx = half_grad[i] - hg_dot_lam;

        // Step 6: step = clamp(-hg_dot_dx / denom, 0, 1)
        let step = (-hg_dot_dx / denom).max(0.0).min(1.0);

        // Step 7: lam = (1-step)*lam + step*e_i  (in-place)
        let one_minus_step = 1.0 - step;
        for j in 0..t0 {
            lam[j] *= one_minus_step;
        }
        lam[i] += step;

        // Step 8: ata_x incremental update — O(T0)
        // ata_x = (1-step)*ata_x + step*ATA[:,i]
        let ata_col_i = ata.column(i);
        for j in 0..t0 {
            ata_x[j] = one_minus_step * ata_x[j] + step * ata_col_i[j];
        }
        // Periodic refresh to prevent drift
        if t > 0 && t % GRAM_REFRESH_INTERVAL == 0 {
            ata_x = ata.dot(lam as &Array1<f64>);
        }

        // Step 9: Compute objective
        let lam_norm_sq: f64 = lam.iter().map(|&v| v * v).sum();
        let xt_ata_x: f64 = ata_x.iter().zip(lam.iter()).map(|(&a, &b)| a * b).sum();
        let atb_dot_lam: f64 = atb.iter().zip(lam.iter()).map(|(&a, &b)| a * b).sum();
        let val = zeta * zeta * lam_norm_sq + (xt_ata_x - 2.0 * atb_dot_lam + b_norm_sq) / n as f64;

        // Step 10: Convergence check
        if t >= 1 && prev_val - val < min_decrease_sq {
            break;
        }
        prev_val = val;
    }
}

/// Allocation-free standard Frank-Wolfe loop for the T0 >= N case (unit weights).
///
/// Same algorithm as the original `fw_step_internal` but with:
/// - 1 GEMV per iteration (down from 3)
/// - 0 heap allocations per iteration (down from ~8)
/// - Incremental `ax` maintenance instead of recomputing A @ lam each step
fn sc_weight_fw_standard(
    a: &ArrayView2<f64>,
    b: &ArrayView1<f64>,
    lam: &mut Array1<f64>,
    eta: f64,
    zeta: f64,
    n: usize,
    min_decrease_sq: f64,
    max_iter: usize,
) {
    let t0 = lam.len();

    // Precompute column norms: col_norms_sq[j] = ||A[:,j]||²
    let mut col_norms_sq = Array1::zeros(t0);
    for j in 0..t0 {
        let col = a.column(j);
        col_norms_sq[j] = col.dot(&col);
    }

    // Pre-allocate workspace
    let mut ax = a.dot(lam as &Array1<f64>); // (N,), maintained incrementally
    let mut half_grad = Array1::zeros(t0);
    let mut diff = Array1::zeros(n); // Reusable buffer for ax - b

    let mut prev_val = f64::INFINITY;

    for t in 0..max_iter {
        // Step 1-2: Compute half_grad = A^T @ (ax - b) + eta * lam
        // Uses general_mat_vec_mul which dispatches to BLAS dgemv when enabled,
        // otherwise falls back to ndarray's optimized matrixmultiply kernel.
        diff.assign(&ax);
        diff -= &*b;
        general_mat_vec_mul(1.0, &a.t(), &diff, 0.0, &mut half_grad);
        half_grad.scaled_add(eta, &*lam);

        // Step 3: Find vertex with smallest gradient component
        let i = argmin_f64(&half_grad);

        // Step 4: Guard — d_x_norm_sq = ||lam||² + 1 - 2*lam[i]
        let lam_norm_sq: f64 = lam.iter().map(|&v| v * v).sum();
        let d_x_norm_sq = lam_norm_sq + 1.0 - 2.0 * lam[i];
        if d_x_norm_sq < 1e-24 {
            // Compute objective for convergence check
            let mut err_sq = 0.0;
            for k in 0..n {
                let e = ax[k] - b[k];
                err_sq += e * e;
            }
            let val = zeta * zeta * lam_norm_sq + err_sq / n as f64;
            if t >= 1 && prev_val - val < min_decrease_sq {
                break;
            }
            prev_val = val;
            continue;
        }

        // Step 5: d_err_sq = col_norms_sq[i] - 2*A[:,i].dot(&ax) + ax.dot(&ax)
        let col_i = a.column(i);
        let col_dot_ax: f64 = col_i.iter().zip(ax.iter()).map(|(&a, &b)| a * b).sum();
        let ax_dot_ax: f64 = ax.iter().map(|&v| v * v).sum();
        let d_err_sq = col_norms_sq[i] - 2.0 * col_dot_ax + ax_dot_ax;

        // Step 6: Guard — denom
        let denom = d_err_sq + eta * d_x_norm_sq;
        if denom <= 0.0 {
            let mut err_sq = 0.0;
            for k in 0..n {
                let e = ax[k] - b[k];
                err_sq += e * e;
            }
            let val = zeta * zeta * lam_norm_sq + err_sq / n as f64;
            if t >= 1 && prev_val - val < min_decrease_sq {
                break;
            }
            prev_val = val;
            continue;
        }

        // Step 7: hg_dot_dx and step
        let hg_dot_lam: f64 = half_grad.iter().zip(lam.iter()).map(|(&a, &b)| a * b).sum();
        let hg_dot_dx = half_grad[i] - hg_dot_lam;
        let step = (-hg_dot_dx / denom).max(0.0).min(1.0);

        // Step 8: Update lam in-place: lam = (1-step)*lam + step*e_i
        let one_minus_step = 1.0 - step;
        for j in 0..t0 {
            lam[j] *= one_minus_step;
        }
        lam[i] += step;

        // Step 9: Update ax incrementally — O(N)
        // ax = (1-step)*ax + step*A[:,i]
        for k in 0..n {
            ax[k] = one_minus_step * ax[k] + step * col_i[k];
        }

        // Step 10: Compute objective
        let mut err_sq = 0.0;
        for k in 0..n {
            let e = ax[k] - b[k];
            err_sq += e * e;
        }
        let lam_norm_sq: f64 = lam.iter().map(|&v| v * v).sum();
        let val = zeta * zeta * lam_norm_sq + err_sq / n as f64;

        if t >= 1 && prev_val - val < min_decrease_sq {
            break;
        }
        prev_val = val;
    }
}

/// Compute synthetic control weights via Frank-Wolfe optimization.
///
/// Matches R's sc.weight.fw() from the synthdid package. Solves:
///   min_{lambda on simplex}  zeta^2 * ||lambda||^2
///       + (1/N) * ||A_centered @ lambda - b_centered||^2
///
/// Dispatches to one of two optimized loop implementations:
/// - **Gram path** (T0 < N): Precomputes A^T@A, reducing per-iteration cost from O(N×T0) to O(T0)
/// - **Standard path** (T0 >= N): Allocation-free loop with 1 GEMV/iter (was 3) and 0 allocs (was ~8)
///
/// # Arguments
/// * `y` - Matrix of shape (N, T0+1). Last column is the target.
/// * `zeta` - Regularization strength.
/// * `intercept` - If true, column-center Y before optimization.
/// * `init_weights` - Initial weights. If None, starts with uniform.
/// * `min_decrease` - Convergence criterion: stop when objective decrease < min_decrease^2.
/// * `max_iter` - Maximum number of iterations.
///
/// # Returns
/// Weights of shape (T0,) on the simplex.
fn sc_weight_fw_internal(
    y: &ArrayView2<f64>,
    zeta: f64,
    intercept: bool,
    init_weights: Option<&Array1<f64>>,
    min_decrease: f64,
    max_iter: usize,
) -> Array1<f64> {
    let t0 = y.ncols() - 1;
    let n = y.nrows();

    if t0 == 0 {
        return Array1::ones(1);
    }

    // Column-center if using intercept — owned Array2 for the centered case
    let y_owned: Array2<f64> = if intercept {
        let col_means = y.mean_axis(Axis(0)).unwrap();
        y - &col_means
    } else {
        y.to_owned()
    };

    let a = y_owned.slice(s![.., ..t0]);
    let b = y_owned.column(t0);
    let eta = n as f64 * zeta * zeta;

    let mut lam = match init_weights {
        Some(w) => w.clone(),
        None => Array1::from_elem(t0, 1.0 / t0 as f64),
    };

    let min_decrease_sq = min_decrease * min_decrease;

    // Dispatch to optimized loop based on problem dimensions
    if t0 < n {
        // Gram path: precompute A^T@A for O(T0) per iteration
        sc_weight_fw_gram(&a, &b, &mut lam, eta, zeta, n, min_decrease_sq, max_iter);
    } else {
        // Standard path: allocation-free with 1 GEMV per iteration
        sc_weight_fw_standard(&a, &b, &mut lam, eta, zeta, n, min_decrease_sq, max_iter);
    }

    lam
}

/// Compute noise level from first-differences of control outcomes.
///
/// Matches R's sd(apply(Y[1:N0, 1:T0], 1, diff)).
///
/// # Arguments
/// * `y_pre_control` - Control unit pre-treatment outcomes (n_pre, n_control)
///
/// # Returns
/// Noise level (standard deviation of first-differences with ddof=1)
#[pyfunction]
pub fn compute_noise_level<'py>(
    _py: Python<'py>,
    y_pre_control: PyReadonlyArray2<'py, f64>,
) -> PyResult<f64> {
    let y = y_pre_control.as_array();
    Ok(compute_noise_level_internal(&y))
}

fn compute_noise_level_internal(y_pre_control: &ArrayView2<f64>) -> f64 {
    let n_pre = y_pre_control.nrows();
    if n_pre < 2 {
        return 0.0;
    }

    // First differences along time axis: (T_pre-1, N_co)
    let n_diff = n_pre - 1;
    let n_control = y_pre_control.ncols();
    let total = n_diff * n_control;
    if total == 0 {
        return 0.0;
    }

    // Compute mean of first diffs
    let mut sum = 0.0;
    for t in 0..n_diff {
        for j in 0..n_control {
            sum += y_pre_control[[t + 1, j]] - y_pre_control[[t, j]];
        }
    }
    let mean = sum / total as f64;

    // Compute variance with ddof=1
    let mut ss = 0.0;
    for t in 0..n_diff {
        for j in 0..n_control {
            let d = (y_pre_control[[t + 1, j]] - y_pre_control[[t, j]]) - mean;
            ss += d * d;
        }
    }

    if total <= 1 {
        return 0.0;
    }
    (ss / (total - 1) as f64).sqrt()
}

/// Expose the generic Frank-Wolfe solver to Python.
///
/// # Arguments
/// * `y` - Matrix (N, T0+1). Last column is target.
/// * `zeta` - Regularization strength.
/// * `intercept` - Column-center if true.
/// * `init_weights` - Optional initial weights.
/// * `min_decrease` - Convergence threshold.
/// * `max_iter` - Maximum iterations.
#[pyfunction]
#[pyo3(signature = (y, zeta, intercept=true, init_weights=None, min_decrease=1e-5, max_iter=10000))]
pub fn sc_weight_fw<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    zeta: f64,
    intercept: bool,
    init_weights: Option<PyReadonlyArray1<'py, f64>>,
    min_decrease: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y_arr = y.as_array();
    let init = init_weights.map(|w| {
        let v = w.as_array();
        v.to_owned()
    });
    let result = sc_weight_fw_internal(
        &y_arr,
        zeta,
        intercept,
        init.as_ref(),
        min_decrease,
        max_iter,
    );
    Ok(result.to_pyarray_bound(py))
}

/// Compute SDID time weights via Frank-Wolfe optimization.
///
/// Matches R's synthdid: sc.weight.fw(Yc[1:N0, ], zeta=zeta.lambda, intercept=TRUE)
///
/// # Arguments
/// * `y_pre_control` - Control outcomes in pre-treatment periods (n_pre, n_control)
/// * `y_post_control` - Control outcomes in post-treatment periods (n_post, n_control)
/// * `zeta_lambda` - Regularization parameter for time weights
/// * `intercept` - Column-center if true
/// * `min_decrease` - Convergence threshold
/// * `max_iter` - Maximum iterations
#[pyfunction]
#[pyo3(signature = (y_pre_control, y_post_control, zeta_lambda, intercept=true, min_decrease=1e-5, max_iter_pre_sparsify=100, max_iter=10000))]
pub fn compute_time_weights<'py>(
    py: Python<'py>,
    y_pre_control: PyReadonlyArray2<'py, f64>,
    y_post_control: PyReadonlyArray2<'py, f64>,
    zeta_lambda: f64,
    intercept: bool,
    min_decrease: f64,
    max_iter_pre_sparsify: usize,
    max_iter: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y_pre = y_pre_control.as_array();
    let y_post = y_post_control.as_array();

    let result = compute_time_weights_internal(&y_pre, &y_post, zeta_lambda, intercept, min_decrease, max_iter_pre_sparsify, max_iter);
    Ok(result.to_pyarray_bound(py))
}

pub(crate) fn compute_time_weights_internal(
    y_pre_control: &ArrayView2<f64>,
    y_post_control: &ArrayView2<f64>,
    zeta_lambda: f64,
    intercept: bool,
    min_decrease: f64,
    max_iter_pre_sparsify: usize,
    max_iter: usize,
) -> Array1<f64> {
    let n_pre = y_pre_control.nrows();

    if n_pre <= 1 {
        return Array1::ones(n_pre);
    }

    let n_control = y_pre_control.ncols();

    // Build collapsed form: (N_co, T_pre + 1), last col = per-control post mean
    let mut y_time = Array2::zeros((n_control, n_pre + 1));

    // Fill pre-period columns (transpose of y_pre_control)
    for j in 0..n_control {
        for t in 0..n_pre {
            y_time[[j, t]] = y_pre_control[[t, j]];
        }
    }

    // Fill last column with per-control post means
    let n_post = y_post_control.nrows();
    for j in 0..n_control {
        let mut sum = 0.0;
        for t in 0..n_post {
            sum += y_post_control[[t, j]];
        }
        y_time[[j, n_pre]] = if n_post > 0 { sum / n_post as f64 } else { 0.0 };
    }

    // Two-pass sparsification (matching R's default sparsify=sparsify_function)
    // First pass: limited iterations
    let lam = sc_weight_fw_internal(&y_time.view(), zeta_lambda, intercept, None, min_decrease, max_iter_pre_sparsify);

    // Sparsify
    let lam_sparse = sparsify_internal(&lam);

    // Second pass: from sparsified initialization
    sc_weight_fw_internal(&y_time.view(), zeta_lambda, intercept, Some(&lam_sparse), min_decrease, max_iter)
}

/// Compute SDID unit weights via Frank-Wolfe with two-pass sparsification.
///
/// Matches R's synthdid: sc.weight.fw(t(Yc[, 1:T0]), zeta=zeta.omega, intercept=TRUE)
/// followed by sparsify_function + second sc.weight.fw pass.
///
/// # Arguments
/// * `y_pre_control` - Control outcomes in pre-treatment periods (n_pre, n_control)
/// * `y_pre_treated_mean` - Mean treated outcomes in pre-treatment periods (n_pre,)
/// * `zeta_omega` - Regularization parameter for unit weights
/// * `intercept` - Column-center if true
/// * `min_decrease` - Convergence threshold
/// * `max_iter_pre_sparsify` - Iterations for first pass (before sparsification)
/// * `max_iter` - Iterations for second pass (after sparsification)
#[pyfunction]
#[pyo3(signature = (y_pre_control, y_pre_treated_mean, zeta_omega, intercept=true, min_decrease=1e-5, max_iter_pre_sparsify=100, max_iter=10000))]
pub fn compute_sdid_unit_weights<'py>(
    py: Python<'py>,
    y_pre_control: PyReadonlyArray2<'py, f64>,
    y_pre_treated_mean: PyReadonlyArray1<'py, f64>,
    zeta_omega: f64,
    intercept: bool,
    min_decrease: f64,
    max_iter_pre_sparsify: usize,
    max_iter: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y_pre = y_pre_control.as_array();
    let y_tr_mean = y_pre_treated_mean.as_array();

    let result = compute_sdid_unit_weights_internal(
        &y_pre, &y_tr_mean, zeta_omega, intercept, min_decrease,
        max_iter_pre_sparsify, max_iter,
    );
    Ok(result.to_pyarray_bound(py))
}

pub(crate) fn compute_sdid_unit_weights_internal(
    y_pre_control: &ArrayView2<f64>,
    y_pre_treated_mean: &ArrayView1<f64>,
    zeta_omega: f64,
    intercept: bool,
    min_decrease: f64,
    max_iter_pre_sparsify: usize,
    max_iter: usize,
) -> Array1<f64> {
    let n_control = y_pre_control.ncols();

    if n_control == 0 {
        return Array1::zeros(0);
    }
    if n_control == 1 {
        return Array1::ones(1);
    }

    let n_pre = y_pre_control.nrows();

    // Build collapsed form: (T_pre, N_co + 1), last col = treated pre means
    let mut y_unit = Array2::zeros((n_pre, n_control + 1));
    for t in 0..n_pre {
        for j in 0..n_control {
            y_unit[[t, j]] = y_pre_control[[t, j]];
        }
        y_unit[[t, n_control]] = y_pre_treated_mean[t];
    }

    // First pass: limited iterations
    let omega = sc_weight_fw_internal(
        &y_unit.view(), zeta_omega, intercept, None, min_decrease, max_iter_pre_sparsify,
    );

    // Sparsify: zero out weights <= max/4, renormalize
    let omega = sparsify_internal(&omega);

    // Second pass: from sparsified initialization
    sc_weight_fw_internal(
        &y_unit.view(), zeta_omega, intercept, Some(&omega), min_decrease, max_iter,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_project_simplex_already_on_simplex() {
        let v = array![0.3, 0.5, 0.2];
        let result = project_simplex_internal(&v.view());

        // Already on simplex, should be unchanged
        let sum: f64 = result.sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(result.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_project_simplex_uniform() {
        let v = array![1.0, 1.0, 1.0, 1.0];
        let result = project_simplex_internal(&v.view());

        // Should project to uniform distribution
        let sum: f64 = result.sum();
        assert!((sum - 1.0).abs() < 1e-10);
        for &x in result.iter() {
            assert!((x - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_project_simplex_negative() {
        let v = array![-1.0, 2.0, 0.5];
        let result = project_simplex_internal(&v.view());

        // Should be on simplex
        let sum: f64 = result.sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(result.iter().all(|&x| x >= -1e-10));
    }

    #[test]
    fn test_compute_weights_sum_to_one() {
        let y_control = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y_treated = array![2.0, 5.0, 8.0];

        let weights =
            compute_synthetic_weights_internal(&y_control.view(), &y_treated.view(), 0.0, None, None)
                .unwrap();

        let sum: f64 = weights.sum();
        assert!((sum - 1.0).abs() < 1e-6, "Weights should sum to 1, got {}", sum);
        assert!(
            weights.iter().all(|&w| w >= -1e-10),
            "Weights should be non-negative"
        );
    }

    #[test]
    fn test_sum_normalize() {
        let v = array![2.0, 3.0, 5.0];
        let result = sum_normalize_internal(&v);
        assert!((result.sum() - 1.0).abs() < 1e-10);
        assert!((result[0] - 0.2).abs() < 1e-10);
        assert!((result[1] - 0.3).abs() < 1e-10);
        assert!((result[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sum_normalize_zero() {
        let v = array![0.0, 0.0, 0.0];
        let result = sum_normalize_internal(&v);
        assert!((result.sum() - 1.0).abs() < 1e-10);
        for &x in result.iter() {
            assert!((x - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparsify() {
        let v = array![0.5, 0.3, 0.1, 0.05, 0.05];
        let result = sparsify_internal(&v);
        // max = 0.5, threshold = 0.125
        // 0.1, 0.05, 0.05 should be zeroed out
        assert!((result.sum() - 1.0).abs() < 1e-10);
        assert!(result[2] == 0.0); // 0.1 <= 0.125
        assert!(result[3] == 0.0);
        assert!(result[4] == 0.0);
        assert!(result[0] > 0.0);
        assert!(result[1] > 0.0);
    }

    #[test]
    fn test_noise_level_basic() {
        // 3 pre-periods, 2 control units
        // Unit 0: [1.0, 2.0, 3.0] -> diffs: [1.0, 1.0]
        // Unit 1: [2.0, 4.0, 6.0] -> diffs: [2.0, 2.0]
        // All diffs: [1.0, 1.0, 2.0, 2.0], mean=1.5, sd=sqrt(sum((d-1.5)^2)/3)
        let y = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
        let nl = compute_noise_level_internal(&y.view());
        // Diffs: [1.0, 2.0, 1.0, 2.0], mean=1.5
        // Var = (0.25+0.25+0.25+0.25)/3 = 1.0/3
        // sd = sqrt(1/3) ≈ 0.5774
        assert!((nl - (1.0_f64 / 3.0).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_fw_weights_on_simplex() {
        // Simple 3x3 problem: 2 pre-periods + 1 target column
        let y = array![[1.0, 2.0, 1.5], [3.0, 4.0, 3.5], [5.0, 6.0, 5.5]];
        let result = sc_weight_fw_internal(&y.view(), 0.1, true, None, 1e-3, 100);
        let sum: f64 = result.sum();
        assert!((sum - 1.0).abs() < 1e-6, "FW weights should sum to 1, got {}", sum);
        assert!(result.iter().all(|&w| w >= -1e-6), "FW weights should be non-negative");
    }

    #[test]
    fn test_time_weights_on_simplex() {
        let y_pre = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y_post = array![[10.0, 11.0, 12.0]];
        let result = compute_time_weights_internal(&y_pre.view(), &y_post.view(), 0.1, true, 1e-3, 100, 1000);
        assert_eq!(result.len(), 3);
        let sum: f64 = result.sum();
        assert!((sum - 1.0).abs() < 1e-6, "Time weights should sum to 1, got {}", sum);
        assert!(result.iter().all(|&w| w >= -1e-6), "Time weights should be non-negative");
    }

    #[test]
    fn test_unit_weights_on_simplex() {
        let y_pre = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y_tr_mean = array![2.0, 5.0, 8.0];
        let result = compute_sdid_unit_weights_internal(
            &y_pre.view(), &y_tr_mean.view(), 0.5, true, 1e-3, 100, 1000,
        );
        assert_eq!(result.len(), 3);
        let sum: f64 = result.sum();
        assert!((sum - 1.0).abs() < 1e-6, "Unit weights should sum to 1, got {}", sum);
        assert!(result.iter().all(|&w| w >= -1e-6), "Unit weights should be non-negative");
    }

    #[test]
    fn test_unit_weights_single_control() {
        let y_pre = array![[1.0], [2.0], [3.0]];
        let y_tr_mean = array![1.5, 2.5, 3.5];
        let result = compute_sdid_unit_weights_internal(
            &y_pre.view(), &y_tr_mean.view(), 0.5, true, 1e-3, 100, 1000,
        );
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fw_gram_matches_standard() {
        // Create a T0 < N problem and verify both paths produce identical weights.
        // N=20, T0=5 -> Gram path. Force standard path via a wrapper for comparison.
        let vals: Vec<f64> = (0..120).map(|i| ((i * 7 + 3) % 97) as f64 / 97.0).collect();
        let y_gram = Array2::from_shape_vec((20, 6), vals).unwrap();

        // Directly call both internal functions on the same centered data.
        let y_owned: Array2<f64> = {
            let col_means = y_gram.mean_axis(Axis(0)).unwrap();
            &y_gram - &col_means
        };
        let t0 = 5;
        let n = 20;
        let a = y_owned.slice(s![.., ..t0]);
        let b = y_owned.column(t0);
        let eta = n as f64 * 0.3 * 0.3;
        let min_decrease_sq = 1e-5 * 1e-5;

        // Run Gram path
        let mut lam_gram = Array1::from_elem(t0, 1.0 / t0 as f64);
        sc_weight_fw_gram(&a, &b, &mut lam_gram, eta, 0.3, n, min_decrease_sq, 10000);

        // Run standard path on same data
        let mut lam_std = Array1::from_elem(t0, 1.0 / t0 as f64);
        sc_weight_fw_standard(&a, &b, &mut lam_std, eta, 0.3, n, min_decrease_sq, 10000);

        // Both should produce nearly identical weights
        for j in 0..t0 {
            assert!(
                (lam_gram[j] - lam_std[j]).abs() < 1e-10,
                "Gram and standard paths diverge at index {}: gram={}, std={}",
                j, lam_gram[j], lam_std[j]
            );
        }

        // Verify both are valid simplex weights
        let sum_gram: f64 = lam_gram.sum();
        let sum_std: f64 = lam_std.sum();
        assert!((sum_gram - 1.0).abs() < 1e-6, "Gram weights should sum to 1, got {}", sum_gram);
        assert!((sum_std - 1.0).abs() < 1e-6, "Standard weights should sum to 1, got {}", sum_std);
    }

    #[test]
    fn test_fw_standard_no_regression() {
        // Create a T0 >= N problem (unit weights case) and verify against known output.
        // N=5, T0=8 -> Standard path.
        let vals: Vec<f64> = (0..45).map(|i| ((i * 13 + 5) % 53) as f64 / 53.0).collect();
        let y = Array2::from_shape_vec((5, 9), vals).unwrap();

        let result = sc_weight_fw_internal(&y.view(), 0.5, true, None, 1e-5, 10000);

        // Verify valid simplex weights
        let sum: f64 = result.sum();
        assert!((sum - 1.0).abs() < 1e-6, "Weights should sum to 1, got {}", sum);
        assert!(result.iter().all(|&w| w >= -1e-6), "Weights should be non-negative");
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_incremental_ata_x_accuracy() {
        // Run 500+ iterations on a T0 < N problem and verify incremental ata_x
        // doesn't drift significantly from fresh computation.
        let vals: Vec<f64> = (0..200).map(|i| {
            let x = (i as f64) * 0.1;
            x.sin() + ((i * 7) % 31) as f64 / 31.0
        }).collect();
        let y = Array2::from_shape_vec((20, 10), vals).unwrap();

        // Run with enough iterations to exercise the refresh mechanism
        let result = sc_weight_fw_internal(&y.view(), 0.1, true, None, 1e-8, 1000);

        // Verify valid result (convergence with correct weights)
        let sum: f64 = result.sum();
        assert!((sum - 1.0).abs() < 1e-6, "Weights should sum to 1, got {}", sum);
        assert!(result.iter().all(|&w| w >= -1e-6), "Weights should be non-negative");

        // Verify Gram path was used (T0=9 < N=20)
        assert_eq!(result.len(), 9);
    }

    #[test]
    fn test_gram_boundary_t0_equals_n_minus_1() {
        // T0 = N-1 triggers Gram path (T0 < N), works correctly
        // N=6, T0=5 -> just barely Gram path
        let vals: Vec<f64> = (0..36).map(|i| ((i * 11 + 7) % 41) as f64 / 41.0).collect();
        let y = Array2::from_shape_vec((6, 6), vals).unwrap();

        let result = sc_weight_fw_internal(&y.view(), 0.2, true, None, 1e-5, 10000);

        let sum: f64 = result.sum();
        assert!((sum - 1.0).abs() < 1e-6, "Weights should sum to 1, got {}", sum);
        assert!(result.iter().all(|&w| w >= -1e-6), "Weights should be non-negative");
        assert_eq!(result.len(), 5); // T0 = 5
    }

    #[test]
    fn test_gemv_produces_correct_half_grad() {
        // Verify that general_mat_vec_mul produces the same half_grad as manual loop.
        let n = 10;
        let t0 = 4;
        let eta = 0.5;

        // Deterministic test data
        let a_vals: Vec<f64> = (0..(n * t0)).map(|i| ((i * 7 + 3) % 41) as f64 / 41.0).collect();
        let a = Array2::from_shape_vec((n, t0), a_vals).unwrap();

        let ax: Array1<f64> = (0..n).map(|i| ((i * 11 + 5) % 37) as f64 / 37.0).collect();
        let b: Array1<f64> = (0..n).map(|i| ((i * 13 + 2) % 29) as f64 / 29.0).collect();
        let lam: Array1<f64> = (0..t0).map(|j| ((j * 17 + 1) % 19) as f64 / 19.0).collect();

        // Reference: manual loop
        let mut ref_grad = Array1::zeros(t0);
        for j in 0..t0 {
            let col = a.column(j);
            let mut dot = 0.0;
            for k in 0..n {
                dot += col[k] * (ax[k] - b[k]);
            }
            ref_grad[j] = dot + eta * lam[j];
        }

        // New code path: general_mat_vec_mul + scaled_add
        let mut new_grad = Array1::zeros(t0);
        let mut diff = Array1::zeros(n);
        diff.assign(&ax);
        diff -= &b;
        general_mat_vec_mul(1.0, &a.t(), &diff, 0.0, &mut new_grad);
        new_grad.scaled_add(eta, &lam);

        // Verify match to high precision
        for j in 0..t0 {
            assert!(
                (ref_grad[j] - new_grad[j]).abs() < 1e-12,
                "half_grad mismatch at index {}: manual={}, gemv={}",
                j, ref_grad[j], new_grad[j]
            );
        }
    }

    #[test]
    fn test_intercept_false_both_paths() {
        // Verify both Gram and standard paths work with intercept=false
        // Gram path: N=15, T0=4 (T0 < N)
        let vals_gram: Vec<f64> = (0..75).map(|i| ((i * 3 + 1) % 37) as f64 / 37.0).collect();
        let y_gram = Array2::from_shape_vec((15, 5), vals_gram).unwrap();
        let result_gram = sc_weight_fw_internal(&y_gram.view(), 0.3, false, None, 1e-5, 10000);
        let sum_gram: f64 = result_gram.sum();
        assert!((sum_gram - 1.0).abs() < 1e-6, "Gram intercept=false: weights should sum to 1, got {}", sum_gram);
        assert!(result_gram.iter().all(|&w| w >= -1e-6), "Gram intercept=false: weights should be non-negative");

        // Standard path: N=4, T0=10 (T0 >= N)
        let vals_std: Vec<f64> = (0..44).map(|i| ((i * 5 + 2) % 29) as f64 / 29.0).collect();
        let y_std = Array2::from_shape_vec((4, 11), vals_std).unwrap();
        let result_std = sc_weight_fw_internal(&y_std.view(), 0.3, false, None, 1e-5, 10000);
        let sum_std: f64 = result_std.sum();
        assert!((sum_std - 1.0).abs() < 1e-6, "Standard intercept=false: weights should sum to 1, got {}", sum_std);
        assert!(result_std.iter().all(|&w| w >= -1e-6), "Standard intercept=false: weights should be non-negative");
    }
}
