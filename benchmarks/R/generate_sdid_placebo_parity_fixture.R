#!/usr/bin/env Rscript
# Generate a fixture pinning R's `synthdid::vcov(method="placebo")` SE plus
# the per-replication permutations R consumed, so the Python R-parity test
# can feed those exact permutations through `_placebo_variance_se` and
# assert SE match at machine precision.
#
# Usage:
#   Rscript benchmarks/R/generate_sdid_placebo_parity_fixture.R
#
# Output:
#   tests/data/sdid_placebo_indices_r.json
#
# Symmetric with the existing jackknife R-parity test
# (TestJackknifeSERParity in tests/test_methodology_sdid.py:1410). Reuses
# the same Y matrix and (N0, N1, T0, T1) shape so the placebo + jackknife
# parity tests share an anchor panel.
#
# R version: 4.5.2; synthdid version: 0.0.9.

library(synthdid)
library(jsonlite)

# Reconstruct R's panel exactly as TestJackknifeSERParity does (set.seed(42),
# 23 units × 8 periods, treated = i > N0 with effect 5 in t > T0).
set.seed(42)
N0 <- 20
N1 <- 3
T0 <- 5
T1 <- 3
N <- N0 + N1
T <- T0 + T1
Y <- matrix(0, nrow = N, ncol = T)
for (i in 1:N) {
  unit_fe <- rnorm(1, sd = 2)
  for (t in 1:T) {
    Y[i, t] <- 10 + unit_fe + (t - 1) * 0.3 + rnorm(1, sd = 0.5)
    if (i > N0 && t > T0) Y[i, t] <- Y[i, t] + 5.0
  }
}

# Fit-time ATT (sanity check — must match TestJackknifeSERParity.R_ATT).
tau_hat <- synthdid_estimate(Y, N0, T0)
r_att <- as.numeric(tau_hat)

# Reproduce R's placebo_se loop exactly so we can record permutations and
# the per-rep tau alongside the resulting SE. Mirrors `synthdid:::placebo_se`
# (R/vcov.R), including the warm-start weights pass-through:
#
#   theta = function(ind) {
#     N0 = length(ind) - N1
#     weights.boot = weights
#     weights.boot$omega = sum_normalize(weights$omega[ind[1:N0]])
#     do.call(synthdid_estimate, c(list(Y = setup$Y[ind, ],
#         N0 = N0, T0 = setup$T0, X = setup$X[ind, , ],
#         weights = weights.boot), opts))
#   }
#
# The warm-start `weights.boot$omega` differs from a fresh uniform init
# at finite FW iterations and is what `vcov(method="placebo")` actually
# consumes — so reproducing it here is required for bit-identical SE.
opts_used <- attr(tau_hat, "opts")
fit_weights <- attr(tau_hat, "weights")
fit_setup <- attr(tau_hat, "setup")
replications <- 200

# Use a fresh seed for the placebo loop so the recorded permutations are
# independent of the fit-time RNG state. Python consumes the recorded
# permutations directly (no RNG-state matching needed).
set.seed(42)
perms <- vector("list", replications)
taus <- numeric(replications)

for (r in 1:replications) {
  ind <- sample(1:N0, N0)
  perms[[r]] <- ind
  N0_placebo <- N0 - N1
  weights_boot <- fit_weights
  weights_boot$omega <- synthdid:::sum_normalize(fit_weights$omega[ind[1:N0_placebo]])
  # IMPORTANT: R's `placebo_se` uses ONLY the N0 controls (subdivided into
  # N0-N1 pseudo-controls + N1 pseudo-treated). Real treated rows are NOT
  # included in the placebo Y matrix — that's what makes the placebo a
  # null-distribution test. ``Y = setup$Y[ind, ]`` is N0 rows; appending
  # the real treated rows (i.e., ``setup$Y[c(ind, (N0+1):N), ]``) would
  # change the test entirely (and produces SE ~0.132 instead of R's 0.226
  # — a 2× drift on this fixture).
  est_placebo <- do.call(
    synthdid_estimate,
    c(list(
      Y = fit_setup$Y[ind, ],
      N0 = N0_placebo,
      T0 = T0,
      X = fit_setup$X[ind, , ],
      weights = weights_boot
    ), opts_used)
  )
  taus[r] <- as.numeric(est_placebo)
}

r_placebo_se <- sqrt((replications - 1) / replications) * sd(taus)

# Sanity check against R's vcov() entry point. With the warm-start pattern
# applied explicitly above, the manual loop and `vcov()` should produce
# the same SE up to MC noise on the seed sequence. Match isn't required
# for the parity test (we use `r_placebo_se` from our recorded
# permutations); both values are kept for transparency.
set.seed(42)
r_placebo_se_via_vcov <- sqrt(vcov(tau_hat, method = "placebo", replications = replications)[1, 1])

cat(sprintf("R ATT:                       %.15f\n", r_att))
cat(sprintf("R placebo SE (manual loop):  %.15f\n", r_placebo_se))
cat(sprintf("R placebo SE (via vcov):     %.15f\n", r_placebo_se_via_vcov))
cat(sprintf("Replications:                %d\n", replications))

# Convert permutations to 0-indexed for Python (R uses 1-indexed).
perms_0indexed <- lapply(perms, function(p) as.integer(p - 1L))

payload <- list(
  metadata = list(
    R_version = paste(R.version$major, R.version$minor, sep = "."),
    synthdid_version = as.character(packageVersion("synthdid")),
    seed = 42L,
    replications = as.integer(replications),
    note = paste(
      "Permutations are 0-indexed for direct numpy consumption.",
      "R ATT, R placebo SE (manual loop), and per-rep taus are pinned",
      "for downstream Python parity assertion."
    )
  ),
  N0 = as.integer(N0),
  N1 = as.integer(N1),
  T0 = as.integer(T0),
  T1 = as.integer(T1),
  R_ATT = r_att,
  R_PLACEBO_SE = r_placebo_se,
  R_PLACEBO_SE_VIA_VCOV = r_placebo_se_via_vcov,
  R_PLACEBO_TAUS = as.numeric(taus),
  R_PERMUTATIONS = perms_0indexed
)

out_path <- "tests/data/sdid_placebo_indices_r.json"
write_json(payload, out_path, auto_unbox = TRUE, digits = 17)
cat(sprintf("\nWrote %s\n", out_path))
