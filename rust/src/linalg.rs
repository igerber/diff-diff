//! Linear algebra operations for OLS estimation and robust variance computation.
//!
//! This module provides optimized implementations of:
//! - OLS solving using LAPACK
//! - HC1 (heteroskedasticity-consistent) variance-covariance estimation
//! - Cluster-robust variance-covariance estimation

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::{FactorizeC, Solve, SolveC, SVD, UPLO};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Solve OLS regression: β = (X'X)^{-1} X'y
///
/// Uses SVD with truncation for rank-deficient matrices:
/// - Computes SVD: X = U * S * V^T
/// - Truncates singular values below rcond * max(S)
/// - Computes solution: β = V * S^{-1}_truncated * U^T * y
///
/// This matches scipy's 'gelsd' driver behavior for handling rank-deficient
/// design matrices that can occur in DiD estimation (e.g., MultiPeriodDiD
/// with redundant period dummies + treatment interactions).
///
/// For rank-deficient matrices (rank < k), the vcov matrix is filled with NaN
/// since the sandwich estimator requires inverting the singular X'X matrix.
/// The Python wrapper should use the full R-style handling with QR pivoting
/// for proper rank-deficiency support.
///
/// # Arguments
/// * `x` - Design matrix (n, k)
/// * `y` - Response vector (n,)
/// * `cluster_ids` - Optional cluster identifiers (n,) as integers
/// * `return_vcov` - Whether to compute and return variance-covariance matrix
///
/// # Returns
/// Tuple of (coefficients, residuals, vcov) where vcov is None if return_vcov=False,
/// or NaN-filled matrix if rank-deficient
#[pyfunction]
#[pyo3(signature = (x, y, cluster_ids=None, return_vcov=true))]
pub fn solve_ols<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    cluster_ids: Option<PyReadonlyArray1<'py, i64>>,
    return_vcov: bool,
) -> PyResult<(
    &'py PyArray1<f64>,
    &'py PyArray1<f64>,
    Option<&'py PyArray2<f64>>,
)> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();

    let n = x_arr.nrows();
    let k = x_arr.ncols();

    // Solve using SVD with truncation for rank-deficient matrices
    // This matches scipy's 'gelsd' behavior
    let x_owned = x_arr.to_owned();
    let y_owned = y_arr.to_owned();

    // Compute SVD: X = U * S * V^T
    let (u_opt, s, vt_opt) = x_owned
        .svd(true, true)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("SVD failed: {}", e)))?;

    let u = u_opt.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("SVD did not return U matrix")
    })?;
    let vt = vt_opt.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("SVD did not return V^T matrix")
    })?;

    // Compute rcond threshold to match R's lm() behavior
    // R's qr() uses tol = 1e-07 by default, which is sqrt(eps) ≈ 1.49e-08
    // We use 1e-07 for consistency with Python backend and R
    let rcond = 1e-07_f64;
    let s_max = s.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = s_max * rcond;

    // Compute truncated pseudoinverse solution: β = V * S^{-1} * U^T * y
    // Singular values below threshold are treated as zero (truncated)
    let uty = u.t().dot(&y_owned); // (min(n,k),)

    // Build S^{-1} with truncation and count effective rank
    let mut s_inv_uty = Array1::<f64>::zeros(k);
    let mut rank = 0usize;
    for i in 0..s.len().min(k) {
        if s[i] > threshold {
            s_inv_uty[i] = uty[i] / s[i];
            rank += 1;
        }
        // else: leave as 0 (truncate this singular value)
    }

    // Compute coefficients: β = V * (S^{-1} * U^T * y)
    let coefficients = vt.t().dot(&s_inv_uty);

    // Compute fitted values and residuals
    let fitted = x_arr.dot(&coefficients);
    let residuals = &y_arr - &fitted;

    // Compute variance-covariance if requested
    // For rank-deficient matrices, return NaN vcov since X'X is singular
    let vcov = if return_vcov {
        if rank < k {
            // Rank-deficient: cannot compute valid vcov, return NaN matrix
            let mut nan_vcov = Array2::<f64>::zeros((k, k));
            nan_vcov.fill(f64::NAN);
            Some(nan_vcov.into_pyarray(py))
        } else {
            // Full rank: compute robust vcov normally
            let cluster_arr = cluster_ids.as_ref().map(|c| c.as_array().to_owned());
            let vcov_arr = compute_robust_vcov_internal(&x_arr, &residuals.view(), cluster_arr.as_ref())?;
            Some(vcov_arr.into_pyarray(py))
        }
    } else {
        None
    };

    Ok((
        coefficients.into_pyarray(py),
        residuals.into_pyarray(py),
        vcov,
    ))
}

/// Compute HC1 or cluster-robust variance-covariance matrix.
///
/// # Arguments
/// * `x` - Design matrix (n, k)
/// * `residuals` - OLS residuals (n,)
/// * `cluster_ids` - Optional cluster identifiers (n,) as integers
///
/// # Returns
/// Variance-covariance matrix (k, k)
#[pyfunction]
#[pyo3(signature = (x, residuals, cluster_ids=None))]
pub fn compute_robust_vcov<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    residuals: PyReadonlyArray1<'py, f64>,
    cluster_ids: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<&'py PyArray2<f64>> {
    let x_arr = x.as_array();
    let residuals_arr = residuals.as_array();
    let cluster_arr = cluster_ids.as_ref().map(|c| c.as_array().to_owned());

    let vcov = compute_robust_vcov_internal(&x_arr, &residuals_arr, cluster_arr.as_ref())?;
    Ok(vcov.into_pyarray(py))
}

/// Internal implementation of robust variance-covariance computation.
fn compute_robust_vcov_internal(
    x: &ArrayView2<f64>,
    residuals: &ArrayView1<f64>,
    cluster_ids: Option<&Array1<i64>>,
) -> PyResult<Array2<f64>> {
    let n = x.nrows();
    let k = x.ncols();

    // Compute X'X
    let xtx = x.t().dot(x);

    // Compute (X'X)^{-1} using Cholesky decomposition
    let xtx_inv = invert_symmetric(&xtx)?;

    match cluster_ids {
        None => {
            // HC1 variance: (X'X)^{-1} X' diag(e²) X (X'X)^{-1} × n/(n-k)
            let u_squared: Array1<f64> = residuals.mapv(|r| r * r);

            // Compute meat = X' diag(e²) X using vectorized BLAS operations
            // This is equivalent to X' @ (X * e²) where e² is broadcast across columns
            // Much faster than O(n*k²) scalar loop - uses optimized BLAS dgemm
            let u_squared_col = u_squared.insert_axis(Axis(1)); // (n, 1)
            let x_weighted = x * &u_squared_col; // (n, k) - broadcasts e² across columns
            let meat = x.t().dot(&x_weighted); // (k, k)

            // HC1 adjustment factor
            let adjustment = n as f64 / (n - k) as f64;

            // Sandwich: (X'X)^{-1} meat (X'X)^{-1}
            let temp = xtx_inv.dot(&meat);
            let vcov = temp.dot(&xtx_inv) * adjustment;

            Ok(vcov)
        }
        Some(clusters) => {
            // Cluster-robust variance
            // Group observations by cluster and sum scores within clusters
            let n_obs = n;

            // Compute scores using vectorized operation: scores = X * residuals[:, np.newaxis]
            // Each row of X is multiplied by its corresponding residual
            let residuals_col = residuals.insert_axis(Axis(1)); // (n, 1)
            let scores = x * &residuals_col; // (n, k) - broadcasts residuals across columns

            // Aggregate scores by cluster using HashMap
            let mut cluster_sums: HashMap<i64, Array1<f64>> = HashMap::new();
            for i in 0..n_obs {
                let cluster = clusters[i];
                let row = scores.row(i).to_owned();
                cluster_sums
                    .entry(cluster)
                    .and_modify(|sum| *sum = &*sum + &row)
                    .or_insert(row);
            }

            let n_clusters = cluster_sums.len();

            if n_clusters < 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Need at least 2 clusters for cluster-robust SEs, got {}", n_clusters)
                ));
            }

            // Build cluster scores matrix (G, k)
            let mut cluster_scores = Array2::<f64>::zeros((n_clusters, k));
            for (idx, (_cluster_id, sum)) in cluster_sums.iter().enumerate() {
                cluster_scores.row_mut(idx).assign(sum);
            }

            // Compute meat: Σ_g (X_g' e_g)(X_g' e_g)'
            let meat = cluster_scores.t().dot(&cluster_scores);

            // Adjustment factors
            // G/(G-1) * (n-1)/(n-k) - matches NumPy implementation
            let g = n_clusters as f64;
            let adjustment = (g / (g - 1.0)) * ((n_obs - 1) as f64 / (n_obs - k) as f64);

            // Sandwich estimator
            let temp = xtx_inv.dot(&meat);
            let vcov = temp.dot(&xtx_inv) * adjustment;

            Ok(vcov)
        }
    }
}

/// Invert a symmetric positive-definite matrix.
///
/// Tries Cholesky factorization first (faster for well-conditioned SPD matrices),
/// falls back to LU decomposition for near-singular or indefinite matrices.
///
/// Cholesky (when applicable):
/// - ~2x faster than LU decomposition
/// - More numerically stable for positive-definite matrices
/// - Reuses the factorization across all column solves
fn invert_symmetric(a: &Array2<f64>) -> PyResult<Array2<f64>> {
    let n = a.nrows();

    // Try Cholesky factorization first (faster for well-conditioned SPD matrices)
    if let Ok(factorized) = a.factorizec(UPLO::Lower) {
        // Solve A X = I for each column using Cholesky
        let mut result = Array2::<f64>::zeros((n, n));
        let mut cholesky_failed = false;

        for i in 0..n {
            let mut e_i = Array1::<f64>::zeros(n);
            e_i[i] = 1.0;

            match factorized.solvec(&e_i) {
                Ok(col) => result.column_mut(i).assign(&col),
                Err(_) => {
                    cholesky_failed = true;
                    break;
                }
            }
        }

        if !cholesky_failed {
            return Ok(result);
        }
    }

    // Fallback to LU decomposition for near-singular or indefinite matrices
    let mut result = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let mut e_i = Array1::<f64>::zeros(n);
        e_i[i] = 1.0;

        let col = a.solve(&e_i).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Matrix inversion failed (likely rank-deficient X'X): {}. \
                 If the design matrix is rank-deficient, use solve_ols without \
                 skip_rank_check=True to enable R-style handling.",
                e
            ))
        })?;

        result.column_mut(i).assign(&col);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_invert_symmetric() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let a_inv = invert_symmetric(&a).unwrap();

        // A * A^{-1} should be identity
        let identity = a.dot(&a_inv);
        assert!((identity[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((identity[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((identity[[0, 1]]).abs() < 1e-10);
        assert!((identity[[1, 0]]).abs() < 1e-10);
    }
}
