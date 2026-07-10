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
use faer::Side;

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
    // This matches scipy's 'gelsd' behavior.
    //
    // Memory discipline (wide designs make one n x k block ~GBs): work from
    // the borrowed numpy views, materialize exactly ONE owned n x k copy
    // (the equilibrated faer matrix below), and drop the SVD factors before
    // the fitted/residual/vcov stage so U never coexists with the vcov
    // scores block. Verified via the alloc-profile counting allocator.

    // Column equilibration: scale each column to unit 2-norm before the SVD so
    // rank detection (threshold = s_max * rcond, anchored to the largest singular
    // value) is invariant to per-column scaling. Without this a column on a large
    // scale (e.g. an unscaled covariate ~1e8) truncates the genuine small-scale
    // direction and returns finite-but-wrong coefficients. Coefficients are
    // unscaled back to raw scale below, BEFORE fitted/residuals/vcov, so all
    // raw-scale quantities (x_arr) stay consistent. Mirrors the Python backend's
    // _detect_rank_deficiency / _equilibrated_lstsq equilibration.
    // Norms are computed from the borrowed view in the same j-outer/i-inner
    // accumulation order as before (bit-identical to the prior owned-copy scan).
    let mut safe_norms = Array1::<f64>::zeros(k);
    for j in 0..k {
        let mut acc = 0.0_f64;
        for i in 0..n {
            let v = x_arr[[i, j]];
            acc += v * v;
        }
        let norm = acc.sqrt();
        safe_norms[j] = if norm > 0.0 { norm } else { 1.0 };
    }

    // Equilibration fused into the faer conversion: the single owned n x k
    // copy. Same per-element `x / norm` as the previous two-step
    // clone-then-divide, so the SVD input values are unchanged.
    let x_faer = faer::Mat::from_fn(n, k, |i, j| x_arr[[i, j]] / safe_norms[j]);

    // Compute thin SVD using faer: X = U * S * V^T
    let svd = match x_faer.thin_svd() {
        Ok(s) => s,
        Err(_) => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "SVD computation failed",
            ))
        }
    };
    // The equilibrated input is not needed once the factors exist.
    drop(x_faer);

    // Extract from the faer SVD result (capitalized methods in faer 0.24)
    // only the small pieces the solve needs - s (min(n,k)), V^T (k x k),
    // and U^T y (min(n,k)) - then drop the factorization, releasing the
    // n x k U storage before any downstream allocation.
    let u_faer = svd.U();
    let s_diag = svd.S(); // Returns diagonal view
    let s_col = s_diag.column_vector(); // Get as column vector
    let v_faer = svd.V(); // This is V, not V^T

    let n_rows = u_faer.nrows();
    let n_svd_cols = u_faer.ncols();

    let s_len = s_col.nrows();
    let mut s = Array1::<f64>::zeros(s_len);
    for i in 0..s_len {
        s[i] = s_col[i]; // S column vector
    }

    let v_rows = v_faer.nrows();
    let v_cols = v_faer.ncols();
    let mut vt = Array2::<f64>::zeros((v_cols, v_rows)); // V^T has shape (k, k)
    for i in 0..v_rows {
        for j in 0..v_cols {
            vt[[j, i]] = v_faer[(i, j)]; // Transpose V to get V^T
        }
    }

    // U^T y computed directly off the faer factor - never materializes an
    // ndarray copy of U. Sequential column-order accumulation (U is
    // col-major): deterministic bits independent of BLAS/thread count.
    let mut uty = Array1::<f64>::zeros(n_svd_cols); // (min(n,k),)
    for j in 0..n_svd_cols {
        let mut acc = 0.0_f64;
        for i in 0..n_rows {
            acc += u_faer[(i, j)] * y_arr[i];
        }
        uty[j] = acc;
    }
    // Everything extracted; release the factorization (frees U's n x k
    // storage) before fitted/residuals/vcov allocate.
    drop(svd);

    // Compute rcond threshold to match R's lm() behavior
    // R's qr() uses tol = 1e-07 by default, which is sqrt(eps) ≈ 1.49e-08
    // We use 1e-07 for consistency with Python backend and R
    let rcond = 1e-07_f64;
    let s_max = s.iter().cloned().fold(0.0_f64, f64::max);
    let threshold = s_max * rcond;

    // Truncated pseudoinverse solution: β = V * S^{-1} * (U^T y).
    // Singular values below threshold are treated as zero (truncated)

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

    // Compute coefficients on the equilibrated design: β_scaled = V * (S^{-1} * U^T * y)
    let mut coefficients = vt.t().dot(&s_inv_uty);

    // Unscale back to raw scale (β = β_scaled / column_norm) BEFORE computing
    // fitted/residuals/vcov, so those operate on the raw x_arr consistently.
    for j in 0..k {
        coefficients[j] /= safe_norms[j];
    }

    // Compute fitted values and residuals (raw x_arr with unscaled coefficients)
    let fitted = x_arr.dot(&coefficients);
    let residuals = &y_arr - &fitted;

    // Compute variance-covariance if requested
    // For rank-deficient matrices, return NaN vcov since X'X is singular
    let vcov = if return_vcov {
        if rank < k {
            // Rank-deficient: cannot compute valid vcov, return NaN matrix
            let mut nan_vcov = Array2::<f64>::zeros((k, k));
            nan_vcov.fill(f64::NAN);
            Some(nan_vcov.to_pyarray(py))
        } else {
            // Full rank: compute robust vcov normally
            let cluster_arr = cluster_ids.as_ref().map(|c| c.as_array().to_owned());
            let vcov_arr = compute_robust_vcov_internal(
                &x_arr,
                &residuals.view(),
                cluster_arr.as_ref(),
                n,
                k,
            )?;
            Some(vcov_arr.to_pyarray(py))
        }
    } else {
        None
    };

    Ok((coefficients.to_pyarray(py), residuals.to_pyarray(py), vcov))
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
    Ok(vcov.to_pyarray(py))
}

/// HC2 (leverage-corrected) heteroskedasticity-robust vcov, one-way only.
///
/// Mirrors the NumPy `_compute_robust_vcov_numpy` unweighted `hc2` branch
/// exactly (sandwich::vcovHC type="HC2" convention):
///   h_i    = x_i' (X'X)^{-1} x_i
///   meat   = X' diag(u_i^2 / max(1 - h_i, 1e-10)) X
///   vcov   = (X'X)^{-1} meat (X'X)^{-1}          (NO n/(n-k) factor)
/// A hat diagonal exceeding 1 + 1e-6 signals a near-singular design; this
/// returns the sentinel error "Hat-matrix diagonal exceeds 1" so the Python
/// dispatcher can reproduce the documented warn-and-fall-back-to-HC1
/// behavior (the guard decision stays in one place, Python-side).
///
/// # Arguments
/// * `x` - Design matrix (n, k)
/// * `residuals` - OLS residuals (n,)
///
/// # Returns
/// Variance-covariance matrix (k, k)
#[pyfunction]
#[pyo3(signature = (x, residuals))]
pub fn compute_robust_vcov_hc2<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    residuals: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_arr = x.as_array();
    let residuals_arr = residuals.as_array();

    if residuals_arr.len() != x_arr.nrows() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "residuals length ({}) must match design rows ({})",
            residuals_arr.len(),
            x_arr.nrows()
        )));
    }

    let xtx = x_arr.t().dot(&x_arr);
    let xtx_inv = invert_symmetric(&xtx)?;

    // Hat diagonals h_i = x_i' (X'X)^{-1} x_i via one GEMM + rowwise dot:
    // H_diag = rowsum((X (X'X)^{-1}) * X).
    let x_bread = x_arr.dot(&xtx_inv); // (n, k)
    let h_diag: Array1<f64> = (&x_bread * &x_arr).sum_axis(Axis(1));

    let h_max = h_diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if h_max > 1.0 + 1e-6 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Hat-matrix diagonal exceeds 1 (max={:.6}); the design is near-singular.",
            h_max
        )));
    }

    // meat = X' diag(u^2 / max(1 - h, 1e-10)) X
    let factor: Array1<f64> = residuals_arr
        .iter()
        .zip(h_diag.iter())
        .map(|(u, h)| u * u / (1.0 - h).max(1e-10))
        .collect();
    let factor_col = factor.insert_axis(Axis(1)); // (n, 1)
    let x_weighted = &x_arr * &factor_col; // (n, k)
    let meat = x_arr.t().dot(&x_weighted); // (k, k)

    // Sandwich WITHOUT DOF adjustment (HC2's leverage correction replaces it).
    let temp = xtx_inv.dot(&meat);
    let vcov = temp.dot(&xtx_inv);
    Ok(vcov.to_pyarray(py))
}

/// Internal implementation of robust variance-covariance computation.
fn compute_robust_vcov_internal(
    x: &ArrayView2<f64>,
    residuals: &ArrayView1<f64>,
    cluster_ids: Option<&Array1<i64>>,
    n: usize,
    k: usize,
) -> PyResult<Array2<f64>> {
    // Saturated design (n <= k): zero residual degrees of freedom make the
    // HC1/CR1 (n - k) adjustment undefined. The documented non-finite
    // inference contract — and the numpy backend, which is canonical (its
    // saturated guard fires before any bread inversion) — is an all-NaN
    // vcov; the previous behavior silently leaked Inf through the n/(n-k)
    // division. Placed BEFORE the bread inversion to mirror the numpy
    // guard order, so direct compute_robust_vcov calls with a singular
    // saturated Gram get the NaN sentinel rather than an inversion error.
    // The G >= 2 cluster contract keeps precedence (evaluated here only in
    // the saturated case, where n is tiny; the general path's in-arm check
    // below is unchanged).
    if n <= k {
        if let Some(clusters) = cluster_ids {
            let (_, n_clusters) = cluster_first_appearance_index(clusters, n);
            if n_clusters < 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Need at least 2 clusters for cluster-robust SEs, got {}",
                    n_clusters
                )));
            }
        }
        let mut nan_vcov = Array2::<f64>::zeros((k, k));
        nan_vcov.fill(f64::NAN);
        return Ok(nan_vcov);
    }

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

            // Aggregate scores by cluster DETERMINISTICALLY. HashMap
            // iteration order is SipHash-randomized per map instance, which
            // reordered the cluster rows on every call — mathematically
            // identical, but the GEMM accumulation order changed, making
            // the clustered vcov run-to-run nondeterministic at ~1e-14
            // (distinct values across identical calls; the Python backend
            // is bit-stable). Rows accumulate in first-appearance order
            // instead: for the factorized 0..G-1 ids the Python dispatcher
            // passes this is ascending id order, matching NumPy's groupby.
            // Index construction is shared with the Cholesky fast-path
            // kernel (cluster_first_appearance_index) so the ordering
            // contract cannot drift between the two vcov paths.
            let (cluster_index, n_clusters) = cluster_first_appearance_index(clusters, n_obs);

            if n_clusters < 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Need at least 2 clusters for cluster-robust SEs, got {}",
                    n_clusters
                )));
            }

            // Build cluster scores matrix (G, k) in first-appearance order
            let mut cluster_scores = Array2::<f64>::zeros((n_clusters, k));
            for i in 0..n_obs {
                let idx = cluster_index[&clusters[i]];
                cluster_scores
                    .row_mut(idx)
                    .zip_mut_with(&scores.row(i), |a, b| *a += *b);
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
pub(crate) fn ndarray_to_faer(arr: &Array2<f64>) -> faer::Mat<f64> {
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
            "Matrix inversion failed (singular matrix)",
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
    let pivot_ratio = if max_pivot > 0.0 {
        min_pivot / max_pivot
    } else {
        0.0
    };

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
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Matrix inversion numerically unstable (residual={:.2e} > threshold={:.2e}). \
                     Design matrix may be near-singular.",
                max_residual, threshold
            )));
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

/// Build the deterministic first-appearance-order cluster index shared by
/// the SVD-path vcov (`compute_robust_vcov_internal`) and the Cholesky
/// fast-path kernel. Returns the id -> row map and the cluster count; the
/// caller enforces its own G >= 2 contract so the two paths raise the same
/// error through their own channels.
fn cluster_first_appearance_index(
    clusters: &Array1<i64>,
    n_obs: usize,
) -> (HashMap<i64, usize>, usize) {
    let mut cluster_index: HashMap<i64, usize> = HashMap::new();
    for i in 0..n_obs {
        let next = cluster_index.len();
        cluster_index.entry(clusters[i]).or_insert(next);
    }
    let n_clusters = cluster_index.len();
    (cluster_index, n_clusters)
}

/// Kernel error that must surface as a Python exception, as opposed to a
/// certification decline (which returns `None` so the dispatcher falls
/// through to the SVD path).
pub(crate) enum CholKernelError {
    /// Cluster-robust vcov requested with fewer than 2 clusters. Message
    /// parity with `compute_robust_vcov_internal` matters: the Python
    /// dispatcher re-raises unless the text mentions instability.
    TooFewClusters(usize),
}

/// Reciprocal-1-norm-condition guard for the Cholesky fast path. Same 1e-6
/// bound as the Python twin's `_SOLVE_OLS_CHOL_RCOND_GUARD` (dpocon):
/// cond(G_eq) <= 1e6 bounds the Cholesky forward error at ~eps*cond
/// ~ 2e-10 relative in the equilibrated basis. The exact 1-norm computed
/// here is >= dpocon's lower-bound estimate, so this rcond is <= the
/// twin's and the guard is marginally STRICTER — a disagreement near the
/// boundary only flips to the verbatim-correct SVD fallback.
const CHOL_RCOND_GUARD: f64 = 1e-6;

/// Output of the certified normal-equations Cholesky solve.
pub(crate) struct CholSolveOutput {
    pub(crate) coefficients: Array1<f64>,
    pub(crate) residuals: Array1<f64>,
    pub(crate) vcov: Option<Array2<f64>>,
}

/// Certified equilibrated normal-equations Cholesky OLS (opt-in fast path).
///
/// Self-certifying: returns `Ok(None)` (the dispatcher then falls through
/// VERBATIM to the SVD `solve_ols`) on any of: k == 0, n < k, zero or
/// non-finite column norm, Llt factorization failure, exact 1-norm
/// rcond(G_eq) <= 1e-6, or a non-finite coefficient. Every heavy operation
/// on this path is faer matmul or a hand-rolled sequential loop — never a
/// BLAS-feature-gated ndarray `.dot()` — so results and performance are
/// uniform across the accelerate/openblas/no-BLAS wheel variants.
///
/// Memory discipline: ONE owned n x k buffer (the equilibrated copy),
/// reused in place for the vcov scores; unlike the SVD path there is no
/// U-factor transient, so the allocator high-water is strictly lower.
///
/// vcov (hc1 / CR1 only, matching the SVD kernel's scope) fuses the bread
/// from the certification byproduct: the rcond computation solves
/// G_eq * Z = I, and (X'X)^{-1} = D^{-1} Z D^{-1} with D = diag(col norms).
/// Adjustment factors, the G >= 2 contract, and the saturated n == k
/// all-NaN vcov contract replicate `compute_robust_vcov_internal`
/// (both paths honor the documented non-finite-inference contract;
/// the SVD path's former silent-Inf leak was fixed alongside this
/// kernel).
pub(crate) fn solve_ols_chol_core(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    cluster_ids: Option<&Array1<i64>>,
    return_vcov: bool,
) -> Result<Option<CholSolveOutput>, CholKernelError> {
    let n = x.nrows();
    let k = x.ncols();
    if k == 0 || n < k {
        return Ok(None);
    }

    // Column 2-norms in the legacy j-outer/i-inner order. Unlike the SVD
    // path (which substitutes 1.0 and lets truncation absorb zero columns),
    // a zero norm here means a structurally deficient design and a
    // non-finite norm means NaN/Inf in X (any non-finite entry poisons its
    // own column's norm) — both are certification declines.
    let mut norms = Array1::<f64>::zeros(k);
    for j in 0..k {
        let mut acc = 0.0_f64;
        for i in 0..n {
            let v = x[[i, j]];
            acc += v * v;
        }
        let norm = acc.sqrt();
        if !norm.is_finite() || !(norm > 0.0) {
            return Ok(None);
        }
        norms[j] = norm;
    }

    // The single owned n x k buffer: equilibrated copy (unit 2-norm
    // columns, entries bounded by 1, so the Gram below is finite by
    // construction given the finite-norm gate above).
    let mut x_eq = faer::Mat::from_fn(n, k, |i, j| x[[i, j]] / norms[j]);

    // Equilibrated Gram and its Cholesky factor. Factorization failure
    // (not numerically PD) declines.
    let gram_eq = x_eq.as_ref().transpose() * x_eq.as_ref();
    let llt = match gram_eq.llt(Side::Lower) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };

    // Certification: exact 1-norm reciprocal condition number via the full
    // inverse Z = G_eq^{-1} (a k x k solve against I — same O(k^3) order as
    // the factorization, microseconds at the k this library sees). Z
    // doubles as the vcov bread below, so certification is not wasted work.
    let identity = faer::Mat::<f64>::from_fn(k, k, |i, j| if i == j { 1.0 } else { 0.0 });
    let z = llt.solve(&identity);
    let mut anorm = 0.0_f64; // ||G_eq||_1
    let mut znorm = 0.0_f64; // ||G_eq^{-1}||_1
    for j in 0..k {
        let mut col_a = 0.0_f64;
        let mut col_z = 0.0_f64;
        for i in 0..k {
            col_a += gram_eq[(i, j)].abs();
            col_z += z[(i, j)].abs();
        }
        anorm = anorm.max(col_a);
        znorm = znorm.max(col_z);
    }
    let denom = anorm * znorm;
    let rcond = if denom > 0.0 && denom.is_finite() {
        1.0 / denom
    } else {
        0.0
    };
    if !(rcond > CHOL_RCOND_GUARD) {
        return Ok(None);
    }

    // beta_eq = G_eq^{-1} (X_eq' y); X_eq'y via sequential column-order
    // accumulation off the col-major faer buffer (deterministic bits, like
    // the SVD path's uty loop).
    let mut xty_eq = faer::Mat::<f64>::zeros(k, 1);
    for j in 0..k {
        let mut acc = 0.0_f64;
        for i in 0..n {
            acc += x_eq[(i, j)] * y[i];
        }
        xty_eq[(j, 0)] = acc;
    }
    let beta_eq = llt.solve(&xty_eq);
    let mut coefficients = Array1::<f64>::zeros(k);
    for j in 0..k {
        let b = beta_eq[(j, 0)] / norms[j];
        if !b.is_finite() {
            return Ok(None);
        }
        coefficients[j] = b;
    }

    // fitted = X beta = X_eq beta_eq exactly (equilibration is a column
    // reparameterization), so the matvec reuses the owned equilibrated
    // buffer through faer instead of a BLAS-gated dot on the raw view.
    let fitted = x_eq.as_ref() * beta_eq.as_ref();
    let mut residuals = Array1::<f64>::zeros(n);
    for i in 0..n {
        residuals[i] = y[i] - fitted[(i, 0)];
    }

    let vcov = if return_vcov {
        // Saturated design (n == k; n < k already declined above): zero
        // residual degrees of freedom, so the sandwich's (n - k) adjustment
        // is undefined. The documented non-finite-inference contract is an
        // all-NaN vcov (matching the numpy saturated guard and the SVD
        // path's guard in compute_robust_vcov_internal).
        // Cluster-contract precedence: the G >= 2 check fires first.
        if let Some(clusters) = cluster_ids {
            let (_, n_clusters) = cluster_first_appearance_index(clusters, n);
            if n_clusters < 2 {
                return Err(CholKernelError::TooFewClusters(n_clusters));
            }
        }
        if n == k {
            let mut nan_vcov = Array2::<f64>::zeros((k, k));
            nan_vcov.fill(f64::NAN);
            return Ok(Some(CholSolveOutput {
                coefficients,
                residuals,
                vcov: Some(nan_vcov),
            }));
        }

        // Overwrite the equilibrated buffer in place with the RAW scores
        // x_ij * u_i (undo the column scaling while folding in the
        // residual) — the memory-floor trick: no second n x k allocation.
        for j in 0..k {
            let nj = norms[j];
            for i in 0..n {
                x_eq[(i, j)] *= nj * residuals[i];
            }
        }

        let (meat, adjustment) = match cluster_ids {
            None => {
                // HC1 meat: S'S with S = diag(u) X.
                let m = x_eq.as_ref().transpose() * x_eq.as_ref();
                (m, n as f64 / (n - k) as f64)
            }
            Some(clusters) => {
                let (cluster_index, n_clusters) = cluster_first_appearance_index(clusters, n);
                // Unreachable when n == k (early return above), but the
                // G >= 2 contract is kept here too for the general path.
                if n_clusters < 2 {
                    return Err(CholKernelError::TooFewClusters(n_clusters));
                }
                let mut cluster_scores = faer::Mat::<f64>::zeros(n_clusters, k);
                for i in 0..n {
                    let idx = cluster_index[&clusters[i]];
                    for j in 0..k {
                        cluster_scores[(idx, j)] += x_eq[(i, j)];
                    }
                }
                let g = n_clusters as f64;
                let adjustment = (g / (g - 1.0)) * ((n - 1) as f64 / (n - k) as f64);
                let m = cluster_scores.as_ref().transpose() * cluster_scores.as_ref();
                (m, adjustment)
            }
        };

        // Bread on the raw scale from the certification byproduct:
        // (X'X)^{-1} = D^{-1} G_eq^{-1} D^{-1}.
        let bread = faer::Mat::<f64>::from_fn(k, k, |i, j| z[(i, j)] / (norms[i] * norms[j]));
        let half = bread.as_ref() * meat.as_ref();
        let sandwich = half.as_ref() * bread.as_ref();
        let mut vcov_arr = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            for j in 0..k {
                vcov_arr[[i, j]] = sandwich[(i, j)] * adjustment;
            }
        }
        Some(vcov_arr)
    } else {
        None
    };

    Ok(Some(CholSolveOutput {
        coefficients,
        residuals,
        vcov,
    }))
}

/// Opt-in certified normal-equations Cholesky OLS (see
/// `solve_ols_chol_core`). Returns Python `None` when certification
/// declines — the dispatcher then falls through verbatim to the SVD
/// `solve_ols` — and the same "Need at least 2 clusters" ValueError as the
/// SVD path's vcov when clustered vcov is requested with G < 2.
///
/// Registered as a SEPARATE pyfunction (not a new parameter on
/// `solve_ols`) so a stale installed extension missing this symbol
/// degrades gracefully via the independent-import pattern in
/// `_backend.py` instead of raising TypeError at call time.
#[pyfunction]
#[pyo3(signature = (x, y, cluster_ids=None, return_vcov=true))]
#[allow(clippy::type_complexity)]
pub fn solve_ols_chol<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    cluster_ids: Option<PyReadonlyArray1<'py, i64>>,
    return_vcov: bool,
) -> PyResult<
    Option<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Option<Bound<'py, PyArray2<f64>>>,
    )>,
> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();
    // Shape contract up front: the Python dispatcher always passes matched
    // shapes, but direct pyfunction misuse should raise a clean ValueError
    // instead of a low-level index panic from the core.
    if y_arr.len() != x_arr.nrows() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "y length ({}) must match X rows ({})",
            y_arr.len(),
            x_arr.nrows()
        )));
    }
    let cluster_arr = cluster_ids.as_ref().map(|c| c.as_array().to_owned());
    if let Some(cl) = cluster_arr.as_ref() {
        if cl.len() != x_arr.nrows() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "cluster_ids length ({}) must match X rows ({})",
                cl.len(),
                x_arr.nrows()
            )));
        }
    }
    match solve_ols_chol_core(&x_arr, &y_arr, cluster_arr.as_ref(), return_vcov) {
        Ok(Some(out)) => Ok(Some((
            out.coefficients.to_pyarray(py),
            out.residuals.to_pyarray(py),
            out.vcov.map(|v| v.to_pyarray(py)),
        ))),
        Ok(None) => Ok(None),
        Err(CholKernelError::TooFewClusters(g)) => {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Need at least 2 clusters for cluster-robust SEs, got {}",
                g
            )))
        }
    }
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
        assert_eq!(
            svd.S().column_vector().nrows(),
            2,
            "S should have min(n,k)=2 singular values"
        );
        assert_eq!(svd.V().nrows(), 3, "V should have k=3 rows");
        assert_eq!(svd.V().ncols(), 2, "V should have min(n,k)=2 cols");

        // Verify s_inv_uty dimension calculation
        let s_len = svd.S().column_vector().nrows();
        assert_eq!(s_len, 2, "s.len() should be min(n,k)=2, not k=3");

        // This is the key fix: s_inv_uty must have dimension s.len()=min(n,k),
        // not k, otherwise vt.t().dot(&s_inv_uty) will have mismatched dimensions
    }

    /// Deterministic pseudo-random design for the chol-kernel tests (no
    /// rand dependency in tests; simple LCG, values in roughly [-1, 1]).
    fn lcg_design(n: usize, k: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
        };
        let mut x = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            x[[i, 0]] = 1.0; // intercept
            for j in 1..k {
                x[[i, j]] = next();
            }
        }
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..k {
                acc += x[[i, j]] * (j as f64 + 0.5);
            }
            y[i] = acc + 0.1 * next();
        }
        (x, y)
    }

    #[test]
    fn test_chol_core_matches_lu_oracle() {
        // Well-conditioned design: chol beta must match the LU
        // normal-equations oracle to ~1e-12 (both exact to ~eps*cond).
        let (x, y) = lcg_design(200, 5, 7);
        let out = solve_ols_chol_core(&x.view(), &y.view(), None, true)
            .ok()
            .flatten()
            .expect("well-conditioned design must certify");

        let xtx = x.t().dot(&x);
        let xty = x.t().dot(&y);
        let xtx_inv = invert_symmetric(&xtx).expect("oracle inverse");
        let beta_oracle = xtx_inv.dot(&xty);
        for j in 0..5 {
            assert!(
                (out.coefficients[j] - beta_oracle[j]).abs() < 1e-12,
                "beta[{}]: chol {} vs oracle {}",
                j,
                out.coefficients[j],
                beta_oracle[j]
            );
        }

        // Bread-from-factor must match the LU-inverse HC1 sandwich to 1e-12.
        let u2: Array1<f64> = out.residuals.mapv(|r| r * r);
        let xw = &x * &u2.insert_axis(ndarray::Axis(1));
        let meat = x.t().dot(&xw);
        let adj = 200.0 / (200.0 - 5.0);
        let vcov_oracle = xtx_inv.dot(&meat).dot(&xtx_inv) * adj;
        let vcov = out.vcov.expect("vcov requested");
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (vcov[[i, j]] - vcov_oracle[[i, j]]).abs() < 1e-12,
                    "vcov[{},{}]: {} vs {}",
                    i,
                    j,
                    vcov[[i, j]],
                    vcov_oracle[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_chol_core_declines_near_singular() {
        // x2 = x1 + 1e-8 * noise: cond(G_eq) far beyond the 1e-6 guard.
        let (x0, y) = lcg_design(300, 2, 11);
        let mut x = Array2::<f64>::zeros((300, 3));
        let mut state = 12345u64;
        for i in 0..300 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
            x[[i, 0]] = x0[[i, 0]];
            x[[i, 1]] = x0[[i, 1]];
            x[[i, 2]] = x0[[i, 1]] + 1e-8 * noise;
        }
        let result = solve_ols_chol_core(&x.view(), &y.view(), None, true)
            .ok()
            .flatten();
        assert!(result.is_none(), "near-singular design must decline");
    }

    #[test]
    fn test_chol_core_declines_on_nan_and_zero_column() {
        let (mut x, y) = lcg_design(50, 3, 13);
        x[[7, 1]] = f64::NAN;
        assert!(
            solve_ols_chol_core(&x.view(), &y.view(), None, false)
                .ok()
                .flatten()
                .is_none(),
            "NaN in X poisons its column norm and must decline"
        );

        let (mut x2, y2) = lcg_design(50, 3, 17);
        for i in 0..50 {
            x2[[i, 2]] = 0.0;
        }
        assert!(
            solve_ols_chol_core(&x2.view(), &y2.view(), None, false)
                .ok()
                .flatten()
                .is_none(),
            "zero column must decline (structural deficiency)"
        );
    }

    #[test]
    fn test_chol_core_declines_underdetermined() {
        let (x, y) = lcg_design(3, 5, 19);
        assert!(
            solve_ols_chol_core(&x.view(), &y.view(), None, false)
                .ok()
                .flatten()
                .is_none(),
            "n < k must decline"
        );
    }

    #[test]
    fn test_chol_core_saturated_design_nan_vcov() {
        // n == k full rank: zero residual df makes the sandwich adjustment
        // undefined; the contract is all-NaN vcov (never Inf). Coefficients
        // and residuals stay finite. Cluster precedence: G < 2 still errors
        // BEFORE the saturated guard.
        let (x, y) = lcg_design(4, 4, 31);
        let out = solve_ols_chol_core(&x.view(), &y.view(), None, true)
            .ok()
            .flatten()
            .expect("saturated full-rank design must certify");
        assert!(out.coefficients.iter().all(|v| v.is_finite()));
        let vcov = out.vcov.expect("vcov requested");
        assert!(
            vcov.iter().all(|v| v.is_nan()),
            "saturated vcov must be all-NaN, got {:?}",
            vcov
        );

        let mut clusters = Array1::<i64>::zeros(4);
        for i in 0..4 {
            clusters[i] = (i % 2) as i64;
        }
        let out2 = solve_ols_chol_core(&x.view(), &y.view(), Some(&clusters), true)
            .ok()
            .flatten()
            .expect("clustered saturated design must certify");
        assert!(out2.vcov.expect("vcov").iter().all(|v| v.is_nan()));

        let one_cluster = Array1::<i64>::zeros(4);
        let res = solve_ols_chol_core(&x.view(), &y.view(), Some(&one_cluster), true);
        assert!(
            matches!(res, Err(CholKernelError::TooFewClusters(1))),
            "G < 2 must take precedence over the saturated guard"
        );
    }

    #[test]
    fn test_chol_core_too_few_clusters_errors() {
        let (x, y) = lcg_design(60, 3, 23);
        let clusters = Array1::<i64>::zeros(60); // single cluster
        let result = solve_ols_chol_core(&x.view(), &y.view(), Some(&clusters), true);
        assert!(
            matches!(result, Err(CholKernelError::TooFewClusters(1))),
            "G < 2 with vcov must error, not decline"
        );
        // ...but with return_vcov=false the cluster contract never fires,
        // matching the SVD path (its check lives inside the vcov build).
        let no_vcov = solve_ols_chol_core(&x.view(), &y.view(), Some(&clusters), false);
        assert!(matches!(no_vcov, Ok(Some(_))));
    }

    #[test]
    fn test_chol_core_clustered_matches_lu_oracle() {
        let (x, y) = lcg_design(120, 4, 29);
        let mut clusters = Array1::<i64>::zeros(120);
        for i in 0..120 {
            clusters[i] = (i % 8) as i64;
        }
        let out = solve_ols_chol_core(&x.view(), &y.view(), Some(&clusters), true)
            .ok()
            .flatten()
            .expect("well-conditioned design must certify");
        let vcov = out.vcov.expect("vcov requested");

        let oracle =
            compute_robust_vcov_internal(&x.view(), &out.residuals.view(), Some(&clusters), 120, 4)
                .expect("oracle vcov");
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (vcov[[i, j]] - oracle[[i, j]]).abs() < 1e-12,
                    "clustered vcov[{},{}]: {} vs {}",
                    i,
                    j,
                    vcov[[i, j]],
                    oracle[[i, j]]
                );
            }
        }
    }

    // Note: Singular and near-singular matrix tests removed because:
    // 1. invert_symmetric() returns PyResult, which requires Python initialization
    //    to create PyErr - `cargo test` without Python causes panic
    // 2. These edge cases are tested at the Python integration level in
    //    tests/test_linalg.py with proper fallback handling
}
