//! Linear algebra operations for OLS estimation and robust variance computation.
//!
//! This module provides optimized implementations of:
//! - OLS solving using pure Rust (faer library)
//! - HC1 (heteroskedasticity-consistent) variance-covariance estimation
//! - Cluster-robust variance-covariance estimation

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use std::collections::HashMap;

// faer for pure Rust linear algebra (no external BLAS/LAPACK dependencies)
use faer::linalg::solvers::{PartialPivLu, Solve};

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
#[allow(clippy::type_complexity)]
pub fn solve_ols<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    cluster_ids: Option<PyReadonlyArray1<'py, i64>>,
    return_vcov: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
)> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();

    let n = x_arr.nrows();
    let k = x_arr.ncols();

    // Solve using SVD with truncation for rank-deficient matrices
    // This matches scipy's 'gelsd' behavior
    let x_owned = x_arr.to_owned();
    let y_owned = y_arr.to_owned();

    // Convert ndarray to faer for SVD computation
    let x_faer = ndarray_to_faer(&x_owned);

    // Compute thin SVD using faer: X = U * S * V^T
    let svd = match x_faer.thin_svd() {
        Ok(s) => s,
        Err(_) => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "SVD computation failed"
            ))
        }
    };

    // Extract U, S, V from faer SVD result (capitalized methods in faer 0.24)
    let u_faer = svd.U();
    let s_diag = svd.S();  // Returns diagonal view
    let s_col = s_diag.column_vector();  // Get as column vector
    let v_faer = svd.V();  // This is V, not V^T

    // Convert back to ndarray
    let n_rows = u_faer.nrows();
    let n_svd_cols = u_faer.ncols();
    let mut u = Array2::<f64>::zeros((n_rows, n_svd_cols));
    for i in 0..n_rows {
        for j in 0..n_svd_cols {
            u[[i, j]] = u_faer[(i, j)];
        }
    }

    let s_len = s_col.nrows();
    let mut s = Array1::<f64>::zeros(s_len);
    for i in 0..s_len {
        s[i] = s_col[i];  // S column vector
    }

    let v_rows = v_faer.nrows();
    let v_cols = v_faer.ncols();
    let mut vt = Array2::<f64>::zeros((v_cols, v_rows));  // V^T has shape (k, k)
    for i in 0..v_rows {
        for j in 0..v_cols {
            vt[[j, i]] = v_faer[(i, j)];  // Transpose V to get V^T
        }
    }

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
    // Note: s.len() = min(n, k) from thin SVD, so this handles underdetermined (n < k) correctly
    let mut s_inv_uty = Array1::<f64>::zeros(s.len());
    let mut rank = 0usize;
    for i in 0..s.len() {
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
            Some(nan_vcov.to_pyarray_bound(py))
        } else {
            // Full rank: compute robust vcov normally
            let cluster_arr = cluster_ids.as_ref().map(|c| c.as_array().to_owned());
            let vcov_arr = compute_robust_vcov_internal(&x_arr, &residuals.view(), cluster_arr.as_ref(), n, k)?;
            Some(vcov_arr.to_pyarray_bound(py))
        }
    } else {
        None
    };

    Ok((
        coefficients.to_pyarray_bound(py),
        residuals.to_pyarray_bound(py),
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
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_arr = x.as_array();
    let residuals_arr = residuals.as_array();
    let cluster_arr = cluster_ids.as_ref().map(|c| c.as_array().to_owned());

    let n = x_arr.nrows();
    let k = x_arr.ncols();
    let vcov = compute_robust_vcov_internal(&x_arr, &residuals_arr, cluster_arr.as_ref(), n, k)?;
    Ok(vcov.to_pyarray_bound(py))
}

/// Internal implementation of robust variance-covariance computation.
fn compute_robust_vcov_internal(
    x: &ArrayView2<f64>,
    residuals: &ArrayView1<f64>,
    cluster_ids: Option<&Array1<i64>>,
    n: usize,
    k: usize,
) -> PyResult<Array2<f64>> {
    // Compute X'X
    let xtx = x.t().dot(x);

    // Compute (X'X)^{-1} using LU decomposition
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

/// Convert ndarray Array2 to faer Mat
fn ndarray_to_faer(arr: &Array2<f64>) -> faer::Mat<f64> {
    let nrows = arr.nrows();
    let ncols = arr.ncols();
    faer::Mat::from_fn(nrows, ncols, |i, j| arr[[i, j]])
}

/// Invert a symmetric positive-definite matrix.
///
/// Uses LU decomposition with partial pivoting. Includes both NaN/Inf check
/// and conditional residual-based verification to catch near-singular matrices
/// that produce finite but numerically inaccurate results.
///
/// Performance optimization: The expensive O(n³) residual check (A * A⁻¹ - I)
/// is only performed when LU pivot ratios suggest potential instability. For
/// well-conditioned matrices (the common case), this check is skipped.
fn invert_symmetric(a: &Array2<f64>) -> PyResult<Array2<f64>> {
    let n = a.nrows();

    // Convert ndarray to faer
    let a_faer = ndarray_to_faer(a);

    // Create identity matrix in faer
    let identity = faer::Mat::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 });

    // Use LU decomposition with partial pivoting
    let lu = PartialPivLu::new(a_faer.as_ref());

    // Solve A * X = I  =>  X = A^{-1}
    let x_faer = lu.solve(&identity);

    // Check for NaN/Inf in result (indicates singular matrix)
    let mut has_nan = false;
    for i in 0..n {
        for j in 0..n {
            if !x_faer[(i, j)].is_finite() {
                has_nan = true;
                break;
            }
        }
        if has_nan {
            break;
        }
    }

    if has_nan {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Matrix inversion failed (singular matrix)"
        ));
    }

    // Check pivot ratio to detect potential instability.
    // The diagonal of U contains the pivots from LU factorization.
    // A small pivot ratio (min/max) indicates potential numerical instability.
    let u_factor = lu.U();
    let mut max_pivot = 0.0_f64;
    let mut min_pivot = f64::INFINITY;
    for i in 0..n {
        let pivot = u_factor[(i, i)].abs();
        if pivot > 0.0 {
            max_pivot = max_pivot.max(pivot);
            min_pivot = min_pivot.min(pivot);
        }
    }
    let pivot_ratio = if max_pivot > 0.0 { min_pivot / max_pivot } else { 0.0 };

    // Only perform expensive residual check if pivots suggest potential instability.
    // Threshold of 1e-10 catches truly problematic matrices while avoiding
    // unnecessary O(n³) computation for well-conditioned cases.
    if pivot_ratio < 1e-10 {
        // Verify inversion accuracy by checking ||A * A^{-1} - I||_max
        // For near-singular matrices, this residual will be large even if
        // the result contains no NaN/Inf values
        let a_times_inv = a_faer.as_ref() * &x_faer;
        let mut max_residual = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let residual = (a_times_inv[(i, j)] - expected).abs();
                max_residual = max_residual.max(residual);
            }
        }

        // Threshold: detect truly singular matrices while allowing ill-conditioned ones
        // Ill-conditioned matrices (high condition number) can have residuals up to ~1e-4
        // while still producing usable results. Use 1e-4 * n as threshold.
        let threshold = 1e-4 * (n as f64);
        if max_residual > threshold {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Matrix inversion numerically unstable (residual={:.2e} > threshold={:.2e}). \
                     Design matrix may be near-singular.",
                    max_residual, threshold
                )
            ));
        }
    }

    // Convert back to ndarray
    let mut result = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = x_faer[(i, j)];
        }
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

    #[test]
    fn test_ndarray_to_faer() {
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let faer_mat = ndarray_to_faer(&arr);
        assert_eq!(faer_mat[(0, 0)], 1.0);
        assert_eq!(faer_mat[(0, 1)], 2.0);
        assert_eq!(faer_mat[(1, 0)], 3.0);
        assert_eq!(faer_mat[(1, 1)], 4.0);
    }

    #[test]
    fn test_svd_underdetermined_dimensions() {
        // Underdetermined system: n=2 observations, k=3 coefficients
        // X is (2, 3), y is (2,)
        // This test verifies that thin SVD returns the correct dimensions
        // for underdetermined systems and that our code handles them correctly
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let _y = array![7.0, 8.0];

        // Convert to faer and compute thin SVD
        let x_faer = ndarray_to_faer(&x);
        let svd = x_faer.thin_svd().unwrap();

        // For n=2 < k=3: U is (2, 2), S has 2 values, V is (3, 2)
        assert_eq!(svd.U().nrows(), 2, "U should have n=2 rows");
        assert_eq!(svd.U().ncols(), 2, "U should have min(n,k)=2 cols");
        assert_eq!(svd.S().column_vector().nrows(), 2, "S should have min(n,k)=2 singular values");
        assert_eq!(svd.V().nrows(), 3, "V should have k=3 rows");
        assert_eq!(svd.V().ncols(), 2, "V should have min(n,k)=2 cols");

        // Verify s_inv_uty dimension calculation
        let s_len = svd.S().column_vector().nrows();
        assert_eq!(s_len, 2, "s.len() should be min(n,k)=2, not k=3");

        // This is the key fix: s_inv_uty must have dimension s.len()=min(n,k),
        // not k, otherwise vt.t().dot(&s_inv_uty) will have mismatched dimensions
    }

    // Note: Singular and near-singular matrix tests removed because:
    // 1. invert_symmetric() returns PyResult, which requires Python initialization
    //    to create PyErr - `cargo test` without Python causes panic
    // 2. These edge cases are tested at the Python integration level in
    //    tests/test_linalg.py with proper fallback handling
}
