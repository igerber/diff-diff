//! Rust backend for diff-diff DiD library.
//!
//! This module provides optimized implementations of computationally
//! intensive operations used in difference-in-differences analysis.

// Pull in BLAS linker flags for macOS Accelerate.
// blas-src is a linker-only crate — extern crate is required to ensure
// the Accelerate framework is actually linked.
// For OpenBLAS (Linux), linking is handled by build.rs instead of blas-src
// to avoid the openblas-src -> ureq -> native-tls dependency chain.
#[cfg(feature = "accelerate")]
extern crate blas_src;

use pyo3::prelude::*;
use std::collections::HashMap;

mod bootstrap;
mod linalg;
mod trop;
mod weights;

/// A Python module implemented in Rust for diff-diff acceleration.
#[pymodule]
fn _rust_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Bootstrap weight generation
    m.add_function(wrap_pyfunction!(
        bootstrap::generate_bootstrap_weights_batch,
        m
    )?)?;

    // Synthetic control weights (legacy projected gradient descent)
    m.add_function(wrap_pyfunction!(weights::compute_synthetic_weights, m)?)?;
    m.add_function(wrap_pyfunction!(weights::project_simplex, m)?)?;

    // SDID weights (Frank-Wolfe matching R's synthdid)
    m.add_function(wrap_pyfunction!(weights::compute_sdid_unit_weights, m)?)?;
    m.add_function(wrap_pyfunction!(weights::compute_time_weights, m)?)?;
    m.add_function(wrap_pyfunction!(weights::compute_noise_level, m)?)?;
    m.add_function(wrap_pyfunction!(weights::sc_weight_fw, m)?)?;

    // Linear algebra operations
    m.add_function(wrap_pyfunction!(linalg::solve_ols, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::compute_robust_vcov, m)?)?;

    // TROP estimator acceleration (twostep method)
    m.add_function(wrap_pyfunction!(trop::compute_unit_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(trop::loocv_grid_search, m)?)?;
    m.add_function(wrap_pyfunction!(trop::bootstrap_trop_variance, m)?)?;

    // TROP estimator acceleration (joint method)
    m.add_function(wrap_pyfunction!(trop::loocv_grid_search_joint, m)?)?;
    m.add_function(wrap_pyfunction!(trop::bootstrap_trop_variance_joint, m)?)?;

    // Diagnostics
    m.add_function(wrap_pyfunction!(rust_backend_info, m)?)?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

/// Return compile-time BLAS feature information for diagnostics.
#[pyfunction]
fn rust_backend_info() -> PyResult<HashMap<String, bool>> {
    let mut info = HashMap::new();
    info.insert(
        "blas".to_string(),
        cfg!(feature = "accelerate") || cfg!(feature = "openblas"),
    );
    info.insert("accelerate".to_string(), cfg!(feature = "accelerate"));
    info.insert("openblas".to_string(), cfg!(feature = "openblas"));
    Ok(info)
}
