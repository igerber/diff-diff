//! Synthetic control weight computation via projected gradient descent.
//!
//! This module provides optimized implementations of:
//! - Synthetic control weight optimization
//! - Simplex projection

use ndarray::{Array1, ArrayView1, ArrayView2};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

/// Maximum number of optimization iterations.
const MAX_ITER: usize = 1000;

/// Default convergence tolerance (matches Python's _OPTIMIZATION_TOL).
const DEFAULT_TOL: f64 = 1e-8;

/// Default step size for gradient descent.
const DEFAULT_STEP_SIZE: f64 = 0.1;

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
}
