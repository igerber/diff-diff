//! Fixed-effects absorption: method-of-alternating-projections demeaning.
//!
//! Exact mirror of the canonical numpy engine `_demean_map_numpy` in
//! `diff_diff/utils.py` (python-canonical policy): same per-column
//! independent convergence loops, same dimension sweep order, the same
//! row-order scatter-add accumulation as `np.bincount`, division by the
//! per-group sums (never multiplication by a reciprocal), and the same
//! `max|x - x_old| < tol` stopping rule with NaN-poisoning semantics (a NaN
//! delta compares false, so a NaN-carrying column never converges - matching
//! numpy's `np.max(...) < tol`). Parallelism is rayon across columns, which
//! preserves the per-variable contract exactly.

use ndarray::{Array1, Array2, ShapeBuilder};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Demean one column in place by alternating projections.
///
/// Returns the 1-based iteration count at convergence, or -1 if the column
/// did not converge within `max_iter` (including the NaN case).
#[allow(clippy::too_many_arguments)]
fn demean_column(
    x: &mut [f64],
    prev: &mut [f64],
    sums: &mut [Vec<f64>],
    codes: &[Vec<usize>],
    denoms: &[Vec<f64>],
    weights: Option<&[f64]>,
    tol: f64,
    max_iter: usize,
) -> i64 {
    let n = x.len();
    for iter in 0..max_iter {
        prev.copy_from_slice(x);
        for (d, codes_d) in codes.iter().enumerate() {
            let sums_d = &mut sums[d];
            sums_d.iter_mut().for_each(|s| *s = 0.0);
            // Row-order scatter-add: the same accumulation order as
            // np.bincount(codes, weights=...).
            match weights {
                Some(w) => {
                    for i in 0..n {
                        sums_d[codes_d[i]] += w[i] * x[i];
                    }
                }
                None => {
                    for i in 0..n {
                        sums_d[codes_d[i]] += x[i];
                    }
                }
            }
            // means = sums / denom; zero-total-weight groups stay at mean 0
            // (rows inert) - the np.divide(where=w_sum > 0) guard semantics.
            let denom_d = &denoms[d];
            for g in 0..sums_d.len() {
                if denom_d[g] > 0.0 {
                    sums_d[g] /= denom_d[g];
                } else {
                    sums_d[g] = 0.0;
                }
            }
            for i in 0..n {
                x[i] -= sums_d[codes_d[i]];
            }
        }
        // all(|delta| < tol) is exactly numpy's max(|delta|) < tol,
        // INCLUDING NaN poisoning: a NaN delta compares false.
        let converged = x.iter().zip(prev.iter()).all(|(a, b)| (a - b).abs() < tol);
        if converged {
            return (iter + 1) as i64;
        }
    }
    -1
}

/// Method-of-alternating-projections demeaning over pre-factorized codes.
///
/// # Arguments
/// * `x` - (n, k) float64 matrix; columns are the variables to demean
/// * `codes` - (n, d) int64 factorized group codes, all in [0, n_groups[j])
/// * `n_groups` - group count per absorbed dimension
/// * `weights` - optional observation weights (weighted group means)
/// * `tol` - convergence tolerance on max|x - x_old| per full sweep cycle
/// * `max_iter` - iteration cap per column
///
/// # Returns
/// Tuple of (demeaned (n, k) float64 array in Fortran order so per-column
/// views are contiguous, iterations-per-column int64 array; -1 marks a
/// column that did not converge).
#[pyfunction]
#[pyo3(signature = (x, codes, n_groups, weights=None, tol=1e-10, max_iter=10_000))]
#[allow(clippy::type_complexity)]
pub fn demean_map<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    codes: PyReadonlyArray2<'py, i64>,
    n_groups: Vec<usize>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    tol: f64,
    max_iter: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>)> {
    let x_arr = x.as_array();
    let codes_arr = codes.as_array();
    let n = x_arr.nrows();
    let k = x_arr.ncols();
    let d = codes_arr.ncols();

    if codes_arr.nrows() != n {
        return Err(PyValueError::new_err(
            "demean_map: x and codes must have the same number of rows",
        ));
    }
    if n_groups.len() != d {
        return Err(PyValueError::new_err(
            "demean_map: n_groups length must match the number of code columns",
        ));
    }
    if n == 0 || k == 0 || d == 0 {
        return Err(PyValueError::new_err(
            "demean_map: empty input (no rows, columns, or dimensions)",
        ));
    }

    // Owned copies FIRST: PyReadonlyArray borrows cannot cross py.detach
    // (compiler-enforced by the Ungil bound). Codes are validated while
    // copying - an out-of-range code would otherwise be a silent
    // wrong-answer scatter.
    let mut codes_owned: Vec<Vec<usize>> = Vec::with_capacity(d);
    for j in 0..d {
        let n_g = n_groups[j];
        let mut col = Vec::with_capacity(n);
        for i in 0..n {
            let c = codes_arr[(i, j)];
            if c < 0 || (c as usize) >= n_g {
                return Err(PyValueError::new_err(format!(
                    "demean_map: code {} out of range [0, {}) in dimension {}",
                    c, n_g, j
                )));
            }
            col.push(c as usize);
        }
        codes_owned.push(col);
    }

    let w_owned: Option<Vec<f64>> = match &weights {
        Some(w) => {
            let wv = w.as_array();
            if wv.len() != n {
                return Err(PyValueError::new_err(
                    "demean_map: weights length must match the number of rows",
                ));
            }
            Some(wv.to_vec())
        }
        None => None,
    };

    // Per-dimension denominators, computed once (counts, or weight sums with
    // the zero-total-weight guard applied at division time).
    let denoms: Vec<Vec<f64>> = codes_owned
        .iter()
        .zip(n_groups.iter())
        .map(|(codes_d, &n_g)| {
            let mut den = vec![0.0f64; n_g];
            match &w_owned {
                Some(w) => {
                    for i in 0..n {
                        den[codes_d[i]] += w[i];
                    }
                }
                None => {
                    for i in 0..n {
                        den[codes_d[i]] += 1.0;
                    }
                }
            }
            den
        })
        .collect();

    // One owned working copy of the data, column-major so each rayon task
    // owns a contiguous column (this buffer becomes the result - no second
    // full-matrix transient).
    let mut cols: Vec<Vec<f64>> = (0..k)
        .map(|j| (0..n).map(|i| x_arr[(i, j)]).collect())
        .collect();

    let w_slice = w_owned.as_deref();
    let codes_ref = &codes_owned;
    let denoms_ref = &denoms;
    let n_groups_ref = &n_groups;

    // Release the GIL for the compute: rayon across columns, each with its
    // own convergence loop and scratch buffers.
    let iters: Vec<i64> = py.detach(|| {
        cols.par_iter_mut()
            .with_min_len(1)
            .map(|col| {
                let mut prev = vec![0.0f64; n];
                let mut sums: Vec<Vec<f64>> =
                    n_groups_ref.iter().map(|&n_g| vec![0.0f64; n_g]).collect();
                demean_column(
                    col, &mut prev, &mut sums, codes_ref, denoms_ref, w_slice, tol, max_iter,
                )
            })
            .collect()
    });

    // Flatten column-major into one buffer (progressively dropping source
    // columns), then expose as a Fortran-order (n, k) array: per-column
    // Python views are contiguous and no transpose copy is needed.
    let mut flat: Vec<f64> = Vec::with_capacity(n * k);
    for col in cols {
        flat.extend_from_slice(&col);
    }
    let out = Array2::from_shape_vec((n, k).f(), flat)
        .map_err(|e| PyValueError::new_err(format!("demean_map: shape error: {}", e)))?;
    let iters_arr = Array1::from(iters);
    Ok((out.into_pyarray(py), iters_arr.into_pyarray(py)))
}

#[cfg(test)]
mod tests {
    use super::demean_column;

    fn run(
        x: &mut Vec<f64>,
        codes: Vec<Vec<usize>>,
        n_groups: Vec<usize>,
        weights: Option<Vec<f64>>,
        tol: f64,
        max_iter: usize,
    ) -> i64 {
        let n = x.len();
        let denoms: Vec<Vec<f64>> = codes
            .iter()
            .zip(n_groups.iter())
            .map(|(codes_d, &n_g)| {
                let mut den = vec![0.0f64; n_g];
                for i in 0..n {
                    den[codes_d[i]] += weights.as_ref().map_or(1.0, |w| w[i]);
                }
                den
            })
            .collect();
        let mut prev = vec![0.0f64; n];
        let mut sums: Vec<Vec<f64>> = n_groups.iter().map(|&g| vec![0.0f64; g]).collect();
        demean_column(
            x,
            &mut prev,
            &mut sums,
            &codes,
            &denoms,
            weights.as_deref(),
            tol,
            max_iter,
        )
    }

    #[test]
    fn one_way_demean_exact() {
        // groups {0: [1, 3], 1: [5, 7]} -> means 2 and 6
        let mut x = vec![1.0, 3.0, 5.0, 7.0];
        let it = run(&mut x, vec![vec![0, 0, 1, 1]], vec![2], None, 1e-12, 100);
        assert!(it >= 1);
        assert_eq!(x, vec![-1.0, 1.0, -1.0, 1.0]);
    }

    #[test]
    fn weighted_zero_weight_group_inert() {
        // group 1 has zero total weight -> its rows keep their values
        let mut x = vec![1.0, 3.0, 5.0, 7.0];
        let it = run(
            &mut x,
            vec![vec![0, 0, 1, 1]],
            vec![2],
            Some(vec![1.0, 1.0, 0.0, 0.0]),
            1e-12,
            100,
        );
        assert!(it >= 1);
        assert_eq!(x, vec![-1.0, 1.0, 5.0, 7.0]);
    }

    #[test]
    fn non_convergence_reports_minus_one() {
        // two-way unbalanced with a starved budget and an impossible tol
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let it = run(
            &mut x,
            vec![vec![0, 0, 1, 1, 2], vec![0, 1, 0, 1, 1]],
            vec![3, 2],
            None,
            0.0, // impossible: |delta| < 0 is never true
            1,
        );
        assert_eq!(it, -1);
    }

    #[test]
    fn nan_in_variable_never_converges() {
        let mut x = vec![1.0, f64::NAN, 3.0, 4.0];
        let it = run(&mut x, vec![vec![0, 0, 1, 1]], vec![2], None, 1e-8, 50);
        assert_eq!(it, -1); // NaN delta compares false -> poisons convergence
    }

    #[test]
    fn two_way_matches_naive_reference() {
        // small unbalanced two-way panel; reference = many-iteration run
        let x0 = vec![2.0, -1.0, 0.5, 3.0, -2.0, 1.0, 4.0];
        let codes = vec![
            vec![0usize, 0, 1, 1, 2, 2, 2],
            vec![0usize, 1, 0, 2, 1, 2, 0],
        ];
        let mut a = x0.clone();
        let it_a = run(&mut a, codes.clone(), vec![3, 3], None, 1e-13, 10_000);
        assert!(it_a >= 1);
        // group means ~0 in every dimension
        for (codes_d, &n_g) in codes.iter().zip([3usize, 3].iter()) {
            let mut sums = vec![0.0f64; n_g];
            let mut cnt = vec![0.0f64; n_g];
            for i in 0..a.len() {
                sums[codes_d[i]] += a[i];
                cnt[codes_d[i]] += 1.0;
            }
            for g in 0..n_g {
                assert!((sums[g] / cnt[g]).abs() < 1e-10);
            }
        }
    }
}
