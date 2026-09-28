#!/usr/bin/env Rscript
# Generate R conleyreg parity golden values for diff-diff Phase 1 Conley SE.
#
# Requires: install.packages("conleyreg")  (CRAN v0.1.9+, plus sf/lwgeom and
#           the system libs gdal/proj/geos/udunits/pkgconf via brew)
# Output:   ../data/r_conleyreg_conley_golden.json
#
# The diff-diff Conley implementation (`diff_diff/linalg.py::_compute_conley_vcov`)
# matches the values in this JSON to atol=1e-6. Earth radius is 6371.01 km
# (matches conleyreg::haversine_dist in src/distance_functions.cpp).

suppressPackageStartupMessages({
  library(conleyreg)
  library(jsonlite)
})

EARTH_RADIUS_KM <- 6371.01  # matches diff-diff and conleyreg

# Build one cross-sectional fixture entry for the JSON. Calls conleyreg with
# lag_cutoff=0 on a singleton-time panel (which collapses to the Phase 1
# cross-sectional sandwich), extracts the vcov, packs everything in the
# schema diff-diff's TestConleyParityR expects.
build_fixture <- function(seed, n, k, metric, cutoff_km, kernel,
                          lat_range, lon_range) {
  set.seed(seed)
  df <- data.frame(
    lat = runif(n, lat_range[1], lat_range[2]),
    lon = runif(n, lon_range[1], lon_range[2])
  )
  for (j in seq_len(k - 1)) df[[paste0("x", j)]] <- rnorm(n)
  betas <- c(1.0, seq(0.5, 2.0, length.out = k - 1))
  df$y <- betas[1] +
    rowSums(sapply(seq_len(k - 1), function(j) betas[j + 1] * df[[paste0("x", j)]])) +
    rnorm(n, sd = 0.5)
  # conleyreg requires unit + time columns even when lag_cutoff=0; supply
  # singleton time series.
  df$unit <- seq_len(n)
  df$time <- 1L

  formula_str <- if (k == 2) "y ~ x1" else
    paste0("y ~ ", paste(paste0("x", seq_len(k - 1)), collapse = " + "))
  # When vcov=TRUE, conleyreg returns the vcov matrix directly (a matrix array).
  V <- conleyreg(
    formula    = as.formula(formula_str),
    data       = df,
    dist_cutoff = cutoff_km,
    unit       = "unit",
    time       = "time",
    lat        = "lat",
    lon        = "lon",
    kernel     = kernel,
    lag_cutoff = 0,
    dist_comp  = if (metric == "haversine") "spherical" else "planar",
    verbose    = FALSE,
    vcov       = TRUE
  )
  V <- unname(as.matrix(V))

  X <- cbind(1, as.matrix(df[, paste0("x", seq_len(k - 1)), drop = FALSE]))
  coords_mat <- as.matrix(df[, c("lat", "lon")])
  # NOTE: R's as.vector on a matrix flattens COLUMN-major; NumPy's reshape
  # reads ROW-major (C order). Transpose first so the flattened vector
  # decodes correctly when np.asarray(...).reshape((n, 2)) is applied.
  list(
    x = as.vector(t(X)),
    x_shape = c(nrow(X), ncol(X)),
    y = df$y,
    coords = as.vector(t(coords_mat)),
    coords_shape = c(n, 2),
    metric = metric,
    cutoff_km = cutoff_km,
    kernel = kernel,
    vcov = as.vector(t(V)),
    vcov_shape = dim(V),
    n = n,
    k = ncol(X)
  )
}

# Build one Phase 2 panel fixture (block-decomposed Conley + per-unit
# Bartlett temporal HAC). conleyreg(lag_cutoff > 0) is the parity benchmark.
# Each unit has a time-invariant (lat, lon) and observations across T periods.
build_panel_fixture <- function(seed, n_units, T_periods, k, cutoff_km, kernel,
                                lag_cutoff, lat_range, lon_range) {
  set.seed(seed)
  lat_unit <- runif(n_units, lat_range[1], lat_range[2])
  lon_unit <- runif(n_units, lon_range[1], lon_range[2])
  unit <- rep(seq_len(n_units), each = T_periods)
  time <- rep(seq_len(T_periods), times = n_units)
  lat <- lat_unit[unit]
  lon <- lon_unit[unit]
  n <- n_units * T_periods
  df <- data.frame(unit = unit, time = time, lat = lat, lon = lon)
  for (j in seq_len(k - 1)) df[[paste0("x", j)]] <- rnorm(n)
  betas <- c(1.0, seq(0.5, 2.0, length.out = k - 1))
  df$y <- betas[1] +
    rowSums(sapply(seq_len(k - 1), function(j) betas[j + 1] * df[[paste0("x", j)]])) +
    rnorm(n, sd = 0.5)

  formula_str <- if (k == 2) "y ~ x1" else
    paste0("y ~ ", paste(paste0("x", seq_len(k - 1)), collapse = " + "))
  V <- conleyreg(
    formula    = as.formula(formula_str),
    data       = df,
    dist_cutoff = cutoff_km,
    unit       = "unit",
    time       = "time",
    lat        = "lat",
    lon        = "lon",
    kernel     = kernel,
    lag_cutoff = lag_cutoff,
    dist_comp  = "spherical",
    verbose    = FALSE,
    vcov       = TRUE
  )
  V <- unname(as.matrix(V))

  X <- cbind(1, as.matrix(df[, paste0("x", seq_len(k - 1)), drop = FALSE]))
  coords_mat <- as.matrix(df[, c("lat", "lon")])
  list(
    x = as.vector(t(X)),
    x_shape = c(nrow(X), ncol(X)),
    y = df$y,
    coords = as.vector(t(coords_mat)),
    coords_shape = c(n, 2),
    unit = df$unit,
    time = df$time,
    metric = "haversine",
    cutoff_km = cutoff_km,
    kernel = kernel,
    lag_cutoff = lag_cutoff,
    vcov = as.vector(t(V)),
    vcov_shape = dim(V),
    n = n,
    k = ncol(X),
    n_units = n_units,
    T_periods = T_periods
  )
}

out <- list(
  meta = list(
    generated_at = format(Sys.Date(), "%Y-%m-%d"),
    earth_radius_km = EARTH_RADIUS_KM,
    tool = paste0(
      "R conleyreg ", as.character(packageVersion("conleyreg")),
      " (Düsterhöft 2021)"
    )
  ),
  # NOTE: only haversine fixtures are anchored against conleyreg. Its planar
  # code path requires a CRS specification (sf object) which is overkill for
  # parity testing — diff-diff's euclidean path is already verified bit-
  # equivalent against scipy.spatial.distance.cdist in
  # tests/test_conley_vcov.py::TestConleyDistanceMetrics::test_pairwise_distance_euclidean_matches_pdist.
  small_haversine = build_fixture(
    seed = 42, n = 50, k = 2, metric = "haversine", cutoff_km = 500,
    kernel = "bartlett", lat_range = c(-30, 30), lon_range = c(-100, 100)
  ),
  dense_haversine = build_fixture(
    seed = 100, n = 200, k = 3, metric = "haversine", cutoff_km = 1000,
    kernel = "bartlett", lat_range = c(-45, 45), lon_range = c(-150, 150)
  ),
  lat_lon_realistic = build_fixture(
    seed = 314, n = 300, k = 3, metric = "haversine", cutoff_km = 200,
    kernel = "bartlett", lat_range = c(25, 50), lon_range = c(-125, -65)
  ),
  # Phase 2 panel fixtures: lag_cutoff > 0 exercises the block-decomposed
  # sandwich (within-period spatial + within-unit Bartlett serial). Each
  # unit's location is time-invariant.
  panel_haversine_lag1 = build_panel_fixture(
    seed = 1001, n_units = 60, T_periods = 3, k = 2,
    cutoff_km = 500, kernel = "bartlett", lag_cutoff = 1,
    lat_range = c(-30, 30), lon_range = c(-100, 100)
  ),
  panel_haversine_lag2 = build_panel_fixture(
    seed = 1002, n_units = 80, T_periods = 5, k = 3,
    cutoff_km = 1000, kernel = "bartlett", lag_cutoff = 2,
    lat_range = c(-45, 45), lon_range = c(-150, 150)
  ),
  panel_lat_lon_realistic_lag1 = build_panel_fixture(
    seed = 1003, n_units = 100, T_periods = 4, k = 3,
    cutoff_km = 200, kernel = "bartlett", lag_cutoff = 1,
    lat_range = c(25, 50), lon_range = c(-125, -65)
  )
)

out_path <- "../data/r_conleyreg_conley_golden.json"
write(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), file = out_path)
cat("Wrote", out_path, "\n")
