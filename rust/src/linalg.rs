//! Linear algebra operations for OLS estimation and robust variance computation.
//!
//! This module provides optimized implementations of:
//! - OLS solving using LAPACK
//! - HC1 (heteroskedasticity-consistent) variance-covariance estimation
//! - Cluster-robust variance-covariance estimation

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::{LeastSquaresSvd, Solve};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Solve OLS regression: β = (X'X)^{-1} X'y
///
/// # Arguments
/// * `x` - Design matrix (n, k)
/// * `y` - Response vector (n,)
/// * `cluster_ids` - Optional cluster identifiers (n,) as integers
/// * `return_vcov` - Whether to compute and return variance-covariance matrix
///
/// # Returns
/// Tuple of (coefficients, residuals, vcov) where vcov is None if return_vcov=False
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

    // Solve least squares using SVD (more stable than normal equations)
    let x_owned = x_arr.to_owned();
    let y_owned = y_arr.to_owned();

    let result = x_owned
        .least_squares(&y_owned)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Least squares failed: {}", e)))?;

    let coefficients = result.solution;

    // Compute fitted values and residuals
    let fitted = x_arr.dot(&coefficients);
    let residuals = &y_arr - &fitted;

    // Compute variance-covariance if requested
    let vcov = if return_vcov {
        let cluster_arr = cluster_ids.as_ref().map(|c| c.as_array().to_owned());
        let vcov_arr = compute_robust_vcov_internal(&x_arr, &residuals.view(), cluster_arr.as_ref())?;
        Some(vcov_arr.into_pyarray(py))
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

            // Compute X' diag(e²) X efficiently
            // meat = Σᵢ eᵢ² xᵢ xᵢ'
            let mut meat = Array2::<f64>::zeros((k, k));
            for i in 0..n {
                let xi = x.row(i);
                let e2 = u_squared[i];
                for j in 0..k {
                    for l in 0..k {
                        meat[[j, l]] += e2 * xi[j] * xi[l];
                    }
                }
            }

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

            // Compute scores: X * e (element-wise, each row multiplied by residual)
            let mut scores = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                let e = residuals[i];
                for j in 0..k {
                    scores[[i, j]] = x[[i, j]] * e;
                }
            }

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
fn invert_symmetric(a: &Array2<f64>) -> PyResult<Array2<f64>> {
    let n = a.nrows();
    let mut result = Array2::<f64>::zeros((n, n));

    // Solve A * x_i = e_i for each column of the identity matrix
    for i in 0..n {
        let mut e_i = Array1::<f64>::zeros(n);
        e_i[i] = 1.0;

        let col = a.solve(&e_i)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Matrix inversion failed: {}", e)))?;

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
