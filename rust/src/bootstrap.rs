//! Bootstrap weight generation for multiplier bootstrap inference.
//!
//! This module provides efficient generation of bootstrap weights
//! using various distributions (Rademacher, Mammen, Webb).

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

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
    // Pre-allocate flat array directly (single allocation instead of Vec<Vec<f64>> + flatten)
    let total_size = n_bootstrap * n_units;
    let mut flat = vec![0.0_f64; total_size];

    // Generate weights in parallel, writing directly to pre-allocated buffer
    flat.par_chunks_mut(n_units)
        .enumerate()
        .for_each(|(i, row)| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(i as u64));
            for val in row.iter_mut() {
                *val = if rng.gen::<bool>() { 1.0 } else { -1.0 };
            }
        });

    Array2::from_shape_vec((n_bootstrap, n_units), flat).unwrap()
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

    // Pre-allocate flat array directly (single allocation)
    let total_size = n_bootstrap * n_units;
    let mut flat = vec![0.0_f64; total_size];

    // Generate weights in parallel, writing directly to pre-allocated buffer
    flat.par_chunks_mut(n_units)
        .enumerate()
        .for_each(|(i, row)| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(i as u64));
            for val in row.iter_mut() {
                *val = if rng.gen::<f64>() < prob_neg {
                    val_neg
                } else {
                    val_pos
                };
            }
        });

    Array2::from_shape_vec((n_bootstrap, n_units), flat).unwrap()
}

/// Generate Webb 6-point distribution weights.
///
/// Six-point distribution that matches additional moments:
/// E[w] = 0, E[w²] = 1, E[w³] = 0, E[w⁴] = 1
///
/// Values: ±√(3/2), ±√(1/2), ±√(1/6) with specific probabilities
fn generate_webb_batch(n_bootstrap: usize, n_units: usize, seed: u64) -> Array2<f64> {
    // Webb 6-point values and cumulative probabilities
    let val1 = (3.0_f64 / 2.0).sqrt(); // √(3/2) ≈ 1.225
    let val2 = (1.0_f64 / 2.0).sqrt(); // √(1/2) ≈ 0.707
    let val3 = (1.0_f64 / 6.0).sqrt(); // √(1/6) ≈ 0.408

    // Equal probability for each of 6 values: 1/6 each
    let prob = 1.0 / 6.0;

    // Pre-allocate flat array directly (single allocation)
    let total_size = n_bootstrap * n_units;
    let mut flat = vec![0.0_f64; total_size];

    // Generate weights in parallel, writing directly to pre-allocated buffer
    flat.par_chunks_mut(n_units)
        .enumerate()
        .for_each(|(i, row)| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(i as u64));
            for val in row.iter_mut() {
                let u = rng.gen::<f64>();
                *val = if u < prob {
                    -val1
                } else if u < 2.0 * prob {
                    -val2
                } else if u < 3.0 * prob {
                    -val3
                } else if u < 4.0 * prob {
                    val3
                } else if u < 5.0 * prob {
                    val2
                } else {
                    val1
                };
            }
        });

    Array2::from_shape_vec((n_bootstrap, n_units), flat).unwrap()
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
