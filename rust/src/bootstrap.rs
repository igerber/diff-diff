//! Bootstrap weight generation for multiplier bootstrap inference.
//!
//! This module provides efficient generation of bootstrap weights
//! using various distributions (Rademacher, Mammen, Webb).

use ndarray::{Array2, Axis};
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Minimum number of bootstrap iterations per parallel task.
/// This reduces scheduling overhead for large n_bootstrap values.
const MIN_CHUNK_SIZE: usize = 64;

/// Generate a batch of bootstrap weights.
///
/// Generates (n_bootstrap, n_units) matrix of bootstrap weights
/// for multiplier bootstrap inference.
///
/// # Arguments
/// * `n_bootstrap` - Number of bootstrap iterations
/// * `n_units` - Number of units (clusters)
/// * `weight_type` - Type of weights: "rademacher", "mammen", or "webb"
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// (n_bootstrap, n_units) array of bootstrap weights
#[pyfunction]
#[pyo3(signature = (n_bootstrap, n_units, weight_type, seed))]
pub fn generate_bootstrap_weights_batch<'py>(
    py: Python<'py>,
    n_bootstrap: usize,
    n_units: usize,
    weight_type: &str,
    seed: u64,
) -> PyResult<&'py PyArray2<f64>> {
    let weights = match weight_type.to_lowercase().as_str() {
        "rademacher" => generate_rademacher_batch(n_bootstrap, n_units, seed),
        "mammen" => generate_mammen_batch(n_bootstrap, n_units, seed),
        "webb" => generate_webb_batch(n_bootstrap, n_units, seed),
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown weight type: {}. Expected 'rademacher', 'mammen', or 'webb'",
                weight_type
            )))
        }
    };

    Ok(weights.into_pyarray(py))
}

/// Generate Rademacher weights: ±1 with equal probability.
///
/// E[w] = 0, Var[w] = 1
fn generate_rademacher_batch(n_bootstrap: usize, n_units: usize, seed: u64) -> Array2<f64> {
    // Pre-allocate output array - eliminates double allocation from Vec<Vec<f64>>
    let mut weights = Array2::<f64>::zeros((n_bootstrap, n_units));

    // Fill rows in parallel using rayon with chunk size tuning
    weights
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .with_min_len(MIN_CHUNK_SIZE)
        .enumerate()
        .for_each(|(i, mut row)| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(i as u64));
            for elem in row.iter_mut() {
                *elem = if rng.gen::<bool>() { 1.0 } else { -1.0 };
            }
        });

    weights
}

/// Generate Mammen weights with two-point distribution.
///
/// w = -(√5 - 1)/2 with probability (√5 + 1)/(2√5)
/// w = (√5 + 1)/2  with probability (√5 - 1)/(2√5)
///
/// E[w] = 0, E[w²] = 1, E[w³] = 1
fn generate_mammen_batch(n_bootstrap: usize, n_units: usize, seed: u64) -> Array2<f64> {
    let sqrt5 = 5.0_f64.sqrt();

    // Two-point distribution values
    let val_neg = -(sqrt5 - 1.0) / 2.0; // ≈ -0.618
    let val_pos = (sqrt5 + 1.0) / 2.0; // ≈ 1.618

    // Probability of negative value
    let prob_neg = (sqrt5 + 1.0) / (2.0 * sqrt5); // ≈ 0.724

    // Pre-allocate output array - eliminates double allocation
    let mut weights = Array2::<f64>::zeros((n_bootstrap, n_units));

    // Fill rows in parallel with chunk size tuning
    weights
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .with_min_len(MIN_CHUNK_SIZE)
        .enumerate()
        .for_each(|(i, mut row)| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(i as u64));
            for elem in row.iter_mut() {
                *elem = if rng.gen::<f64>() < prob_neg {
                    val_neg
                } else {
                    val_pos
                };
            }
        });

    weights
}

/// Generate Webb 6-point distribution weights.
///
/// Six-point distribution that matches additional moments:
/// E[w] = 0, E[w²] = 1, E[w³] = 0, E[w⁴] = 1
///
/// Values: ±√(3/2), ±√(1/2), ±√(1/6) with specific probabilities
fn generate_webb_batch(n_bootstrap: usize, n_units: usize, seed: u64) -> Array2<f64> {
    // Webb 6-point values
    let val1 = (3.0_f64 / 2.0).sqrt(); // √(3/2) ≈ 1.225
    let val2 = (1.0_f64 / 2.0).sqrt(); // √(1/2) ≈ 0.707
    let val3 = (1.0_f64 / 6.0).sqrt(); // √(1/6) ≈ 0.408

    // Lookup table for direct index computation (replaces 6-way if-else)
    // Equal probability: u in [0, 1/6) -> -val1, [1/6, 2/6) -> -val2, etc.
    let weights_table = [-val1, -val2, -val3, val3, val2, val1];

    // Pre-allocate output array - eliminates double allocation
    let mut weights = Array2::<f64>::zeros((n_bootstrap, n_units));

    // Fill rows in parallel with chunk size tuning
    weights
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .with_min_len(MIN_CHUNK_SIZE)
        .enumerate()
        .for_each(|(i, mut row)| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(i as u64));
            for elem in row.iter_mut() {
                let u = rng.gen::<f64>();
                // Direct bucket computation: multiply by 6 and floor to get index 0-5
                // Clamp to 5 to handle edge case where u == 1.0
                let bucket = ((u * 6.0).floor() as usize).min(5);
                *elem = weights_table[bucket];
            }
        });

    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rademacher_shape() {
        let weights = generate_rademacher_batch(100, 50, 42);
        assert_eq!(weights.shape(), &[100, 50]);
    }

    #[test]
    fn test_rademacher_values() {
        let weights = generate_rademacher_batch(10, 100, 42);

        for w in weights.iter() {
            assert!(*w == 1.0 || *w == -1.0, "Rademacher weight should be ±1");
        }
    }

    #[test]
    fn test_rademacher_mean_approx_zero() {
        let weights = generate_rademacher_batch(1000, 1, 42);
        let mean: f64 = weights.iter().sum::<f64>() / weights.len() as f64;

        // With 1000 samples, mean should be close to 0
        assert!(
            mean.abs() < 0.1,
            "Rademacher mean should be close to 0, got {}",
            mean
        );
    }

    #[test]
    fn test_mammen_shape() {
        let weights = generate_mammen_batch(100, 50, 42);
        assert_eq!(weights.shape(), &[100, 50]);
    }

    #[test]
    fn test_mammen_mean_approx_zero() {
        let weights = generate_mammen_batch(1000, 1, 42);
        let mean: f64 = weights.iter().sum::<f64>() / weights.len() as f64;

        assert!(
            mean.abs() < 0.1,
            "Mammen mean should be close to 0, got {}",
            mean
        );
    }

    #[test]
    fn test_webb_shape() {
        let weights = generate_webb_batch(100, 50, 42);
        assert_eq!(weights.shape(), &[100, 50]);
    }

    #[test]
    fn test_reproducibility() {
        let weights1 = generate_rademacher_batch(100, 50, 42);
        let weights2 = generate_rademacher_batch(100, 50, 42);

        // Same seed should produce same results
        assert_eq!(weights1, weights2);
    }

    #[test]
    fn test_different_seeds() {
        let weights1 = generate_rademacher_batch(100, 50, 42);
        let weights2 = generate_rademacher_batch(100, 50, 43);

        // Different seeds should produce different results
        assert_ne!(weights1, weights2);
    }
}
