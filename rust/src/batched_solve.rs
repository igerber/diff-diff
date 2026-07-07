//! Batched ridge-regularized SPD solves for EfficientDiD per-unit weights.
//!
//! Solves `(A_i + ridge_i * I) x_i = 1` for a stack of small symmetric
//! matrices (H = 2..~60), one solve per unit per (g, t) cell. The matrices
//! are numerically PSD by construction (conditional Omega* covariances) and
//! the trace-scaled ridge makes them SPD, so an unblocked Cholesky
//! factorization (1/3 H^3 flops vs LU's 2/3 H^3) in a reused per-thread
//! scratch buffer is the primary path, parallelized over the batch with
//! rayon. Each matrix is solved independently in a fixed operation order, so
//! results are bit-deterministic across thread counts and batch splits (the
//! Python tile-invariance contract depends on this).
//!
//! The GIL is held throughout (PyReadonlyArray borrows cannot cross
//! `py.detach`; same pattern as `trop.rs`). Nothing in the parallel region
//! touches Python.

use ndarray::{Array2, ArrayView2, ArrayViewMut1, Axis};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use faer::linalg::solvers::{PartialPivLu, Solve};

/// Solve `(A_i + ridge_i * I) x_i = 1` for a stack of symmetric matrices.
///
/// # Arguments
/// * `a_stack` - (m, H, H) stack of symmetric (numerically PSD) matrices
/// * `ridge` - (m,) per-matrix ridge added to the diagonal (single addition
///   per diagonal entry - bit-identical to numpy's `a + r * I` there)
///
/// # Returns
/// (m, H) solutions. Rows where the matrix is not SPD fall back to a faer
/// partial-pivot LU solve; a row whose LU has an exactly-zero pivot is
/// poisoned with NaN (mirroring LAPACK dgesv's exact-singularity signal so
/// the Python caller can route those rows through its legacy pinv path).
/// Non-finite values from a near-singular LU solve are returned as-is - the
/// Python caller treats any non-finite row as "recompute via legacy path".
///
/// Degenerate shapes: m == 0 -> (0, H); H == 0 -> (m, 0). H == 1 solves via
/// the same Cholesky code (x = 1/sqrt(d)/sqrt(d), within 1 ulp of 1/d).
#[pyfunction]
#[pyo3(signature = (a_stack, ridge))]
pub fn batched_ridge_chol_solve_ones<'py>(
    py: Python<'py>,
    a_stack: PyReadonlyArray3<'py, f64>,
    ridge: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = a_stack.as_array();
    let r = ridge.as_array();
    let m = a.shape()[0];
    let h = a.shape()[1];

    if a.shape()[2] != h {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "a_stack must be a stack of square matrices, got ({}, {}, {})",
            m,
            h,
            a.shape()[2]
        )));
    }
    if r.len() != m {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "ridge length {} must match batch size {}",
            r.len(),
            m
        )));
    }

    let mut out = Array2::<f64>::zeros((m, h));
    if m == 0 || h == 0 {
        return Ok(out.to_pyarray(py));
    }

    // Bound per-task scheduling overhead: tiny matrices (H=2) get large
    // chunks, big matrices (H=60, ~10us each) split freely.
    let min_len = (4096 / (h * h)).max(1);

    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .zip(a.axis_iter(Axis(0)).into_par_iter())
        .with_min_len(min_len)
        .for_each_init(
            || (vec![0.0_f64; h * h], vec![0.0_f64; h]),
            |(buf, x), ((i, mut out_row), a_i)| {
                solve_one(a_i, r[i], buf, x, &mut out_row);
            },
        );

    Ok(out.to_pyarray(py))
}

/// Solve one (H, H) system into `out_row`, using the per-thread scratch.
fn solve_one(
    a_i: ArrayView2<'_, f64>,
    ridge_i: f64,
    buf: &mut [f64],
    x: &mut [f64],
    out_row: &mut ArrayViewMut1<'_, f64>,
) {
    let h = x.len();

    // Copy into row-major scratch, adding the ridge on the diagonal. Every
    // entry of buf is overwritten, so no state leaks between rows.
    for row in 0..h {
        for col in 0..h {
            buf[row * h + col] = a_i[[row, col]];
        }
        buf[row * h + row] += ridge_i;
    }

    if cholesky_solve_ones_in_place(buf, x, h) {
        for (o, v) in out_row.iter_mut().zip(x.iter()) {
            *o = *v;
        }
    } else {
        lu_fallback(a_i, ridge_i, out_row, h);
    }
}

/// Unblocked lower Cholesky in place on `buf` (row-major, lower triangle
/// written; upper triangle left as input values and never read), then
/// forward/back substitution against the implicit ones RHS into `x`.
/// Returns false without touching `x` if a pivot is not strictly positive.
// The negated comparison is the point: `!(d > 0.0)` is true for NaN while
// `d <= 0.0` is not, and NaN pivots must route to the LU fallback.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn cholesky_solve_ones_in_place(buf: &mut [f64], x: &mut [f64], h: usize) -> bool {
    for j in 0..h {
        let mut d = buf[j * h + j];
        for k in 0..j {
            d -= buf[j * h + k] * buf[j * h + k];
        }
        // NaN-safe pivot check: must be `!(d > 0.0)`, NOT `d <= 0.0` - a NaN
        // pivot compares false either way, and only this form routes NaN to
        // the LU fallback instead of silently continuing with sqrt(NaN).
        if !(d > 0.0) {
            return false;
        }
        let l_jj = d.sqrt();
        buf[j * h + j] = l_jj;
        for row in (j + 1)..h {
            let mut v = buf[row * h + j];
            for k in 0..j {
                v -= buf[row * h + k] * buf[j * h + k];
            }
            buf[row * h + j] = v / l_jj;
        }
    }

    // Forward substitution: L y = 1.
    for row in 0..h {
        let mut v = 1.0_f64;
        for k in 0..row {
            v -= buf[row * h + k] * x[k];
        }
        x[row] = v / buf[row * h + row];
    }

    // Back substitution: L^T z = y (reads L transposed).
    for row in (0..h).rev() {
        let mut v = x[row];
        for k in (row + 1)..h {
            v -= buf[k * h + row] * x[k];
        }
        x[row] = v / buf[row * h + row];
    }

    true
}

/// Rare-path fallback for non-SPD matrices: faer partial-pivot LU.
///
/// An exactly-zero U pivot poisons the row with NaN - LAPACK's dgesv raises
/// on exact zero pivots (numpy's LinAlgError -> legacy pinv route), while a
/// different elimination order could otherwise produce finite garbage here.
fn lu_fallback(
    a_i: ArrayView2<'_, f64>,
    ridge_i: f64,
    out_row: &mut ArrayViewMut1<'_, f64>,
    h: usize,
) {
    let a_faer = faer::Mat::from_fn(h, h, |row, col| {
        let v = a_i[[row, col]];
        if row == col {
            v + ridge_i
        } else {
            v
        }
    });
    let lu = PartialPivLu::new(a_faer.as_ref());

    let u = lu.U();
    for k in 0..h {
        if u[(k, k)] == 0.0 {
            for o in out_row.iter_mut() {
                *o = f64::NAN;
            }
            return;
        }
    }

    let ones = faer::Mat::from_fn(h, 1, |_, _| 1.0);
    let sol = lu.solve(&ones);
    for (k, o) in out_row.iter_mut().enumerate() {
        *o = sol[(k, 0)];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// Random SPD matrix A = B B^T + eps * I.
    fn random_spd(h: usize, rng: &mut Xoshiro256PlusPlus, eps: f64) -> ndarray::Array2<f64> {
        let b = ndarray::Array2::from_shape_fn((h, h), |_| rng.random::<f64>() - 0.5);
        let mut a = b.dot(&b.t());
        for j in 0..h {
            a[[j, j]] += eps;
        }
        a
    }

    /// Drive solve_one directly (the pyfunction wrapper needs Python).
    fn solve_stack(a: &Array3<f64>, ridge: &Array1<f64>) -> Array2<f64> {
        let m = a.shape()[0];
        let h = a.shape()[1];
        let mut out = Array2::<f64>::zeros((m, h));
        let mut buf = vec![0.0_f64; h * h];
        let mut x = vec![0.0_f64; h];
        for (i, mut row) in out.axis_iter_mut(Axis(0)).enumerate() {
            solve_one(
                a.index_axis(Axis(0), i),
                ridge[i],
                &mut buf,
                &mut x,
                &mut row,
            );
        }
        out
    }

    /// Oracle: faer LU on the same ridged matrix.
    fn lu_oracle(a: &ndarray::Array2<f64>, ridge: f64) -> Vec<f64> {
        let h = a.nrows();
        let a_faer = faer::Mat::from_fn(
            h,
            h,
            |r, c| {
                if r == c {
                    a[[r, c]] + ridge
                } else {
                    a[[r, c]]
                }
            },
        );
        let lu = PartialPivLu::new(a_faer.as_ref());
        let ones = faer::Mat::from_fn(h, 1, |_, _| 1.0);
        let sol = lu.solve(&ones);
        (0..h).map(|k| sol[(k, 0)]).collect()
    }

    #[test]
    fn cholesky_matches_lu_oracle_on_random_spd() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        for &h in &[1usize, 2, 3, 8, 30, 60] {
            let m = 5;
            let mut a = Array3::<f64>::zeros((m, h, h));
            for i in 0..m {
                a.index_axis_mut(Axis(0), i)
                    .assign(&random_spd(h, &mut rng, 0.5));
            }
            let ridge = Array1::from_elem(m, 1e-6);
            let out = solve_stack(&a, &ridge);
            for i in 0..m {
                let oracle = lu_oracle(&a.index_axis(Axis(0), i).to_owned(), ridge[i]);
                for k in 0..h {
                    let got = out[[i, k]];
                    let want = oracle[k];
                    let rel = (got - want).abs() / want.abs().max(1e-30);
                    assert!(
                        rel < 1e-10,
                        "h={} i={} k={} got={} want={} rel={}",
                        h,
                        i,
                        k,
                        got,
                        want,
                        rel
                    );
                }
            }
        }
    }

    #[test]
    fn nan_input_row_produces_non_finite_output() {
        let mut a = Array3::<f64>::zeros((1, 3, 3));
        for j in 0..3 {
            a[[0, j, j]] = 1.0;
        }
        a[[0, 1, 1]] = f64::NAN;
        let ridge = Array1::from_elem(1, 1e-6);
        let out = solve_stack(&a, &ridge);
        assert!(
            out.iter().any(|v| !v.is_finite()),
            "NaN input must not produce an all-finite row: {:?}",
            out
        );
    }

    #[test]
    fn exact_singular_zero_ridge_row_is_nan_poisoned() {
        // diag(1, -1, 0): trace 0 -> zero ridge upstream; Cholesky fails at
        // the -1 pivot; LU has an exactly-zero pivot -> poisoned row.
        let mut a = Array3::<f64>::zeros((1, 3, 3));
        a[[0, 0, 0]] = 1.0;
        a[[0, 1, 1]] = -1.0;
        a[[0, 2, 2]] = 0.0;
        let ridge = Array1::from_elem(1, 0.0);
        let out = solve_stack(&a, &ridge);
        assert!(
            out.iter().all(|v| v.is_nan()),
            "exact-singular row must be NaN-poisoned: {:?}",
            out
        );
    }

    #[test]
    fn indefinite_full_rank_row_uses_lu_exactly() {
        // diag(1, -1): Cholesky fails; LU gives exactly [1, -1].
        let mut a = Array3::<f64>::zeros((1, 2, 2));
        a[[0, 0, 0]] = 1.0;
        a[[0, 1, 1]] = -1.0;
        let ridge = Array1::from_elem(1, 0.0);
        let out = solve_stack(&a, &ridge);
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], -1.0);
    }

    #[test]
    fn h_equals_one_matches_reciprocal() {
        let mut a = Array3::<f64>::zeros((1, 1, 1));
        a[[0, 0, 0]] = 4.0;
        let ridge = Array1::from_elem(1, 0.0);
        let out = solve_stack(&a, &ridge);
        let rel = (out[[0, 0]] - 0.25).abs() / 0.25;
        assert!(rel < 1e-14, "got {}", out[[0, 0]]);
    }
}
