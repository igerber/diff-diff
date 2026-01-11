//! Rust backend for diff-diff DiD library.
//!
//! This module provides optimized implementations of computationally
//! intensive operations used in difference-in-differences analysis.

use pyo3::prelude::*;

mod bootstrap;
mod linalg;
mod weights;

/// A Python module implemented in Rust for diff-diff acceleration.
#[pymodule]
fn _rust_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    // Bootstrap weight generation
    m.add_function(wrap_pyfunction!(
        bootstrap::generate_bootstrap_weights_batch,
        m
    )?)?;

    // Synthetic control weights
    m.add_function(wrap_pyfunction!(weights::compute_synthetic_weights, m)?)?;
    m.add_function(wrap_pyfunction!(weights::project_simplex, m)?)?;

    // Linear algebra operations
    m.add_function(wrap_pyfunction!(linalg::solve_ols, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::compute_robust_vcov, m)?)?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
