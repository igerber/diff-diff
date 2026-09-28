#!/usr/bin/env Rscript
# Golden generator: wild_bootstrap_se vs R `fwildclusterboot::boottest`
# (Roodman, MacKinnon, Nielsen & Webb 2019; Cameron-Gelbach-Miller 2008).
#
# Writes a fixed few-cluster 2x2 DiD design and the R reference so that
# tests/test_wild_bootstrap.py::TestWildBootstrapParityR can pin Python output
# against R without requiring R at test time.
#
# Outputs (checked into the repo):
#   benchmarks/data/wild_cluster_boot_test_data.csv  (cluster, treated, post, y)
#   benchmarks/data/wild_cluster_boot_golden.json
#
# Usage:
#   Rscript benchmarks/R/generate_wild_cluster_boot_golden.R
#
# Notes:
#   - boottest defaults are the WCR (restricted) bootstrap: impose_null=TRUE,
#     type="rademacher". boottest switches to full enumeration once B reaches the
#     number of possible draws (2^G); with G=6 and B=99999 it enumerates all
#     2^6=64 raw sign-vectors (deterministic, no RNG), of which 2^(G-1)=32 have
#     distinct |t|. The library's wild_bootstrap_se uses the same trigger
#     (2^n_clusters <= n_bootstrap) and also materializes all 2^G raw vectors,
#     reporting n_bootstrap = 2^G; both yield the same p-value and CI.
#   - The DGP uses a deliberately weak effect + heavy noise so the bootstrap
#     p-value is INTERIOR (not 0/1), letting the test pin the exact p-value.
#     boottest counts strict exceedances |t*| > |t0|; the library matches this
#     (it floors the reported p at 1/(B+1), inactive for an interior p).
#   - se is the analytical CR1 cluster-robust SE = (G/(G-1))((N-1)/(N-k)) sandwich,
#     which the library reports as `se` and uses to studentize the test. boottest
#     studentizes with the same CR1 SE, so teststat == coef/se.
#   - The CI is obtained by inverting the bootstrap test (boottest's grid search
#     vs the library's bisection); they agree to ~1e-4 on this design.

suppressMessages({
  library(fwildclusterboot)
  library(fixest)
  library(jsonlite)
})

set.seed(20240624)
G <- 6
obs_per_cluster <- 8
rows <- list()
i <- 1
for (c in 0:(G - 1)) {
  is_treated <- as.integer(c < G / 2)
  cluster_effect <- rnorm(1, 0, 1.5)
  for (o in 1:obs_per_cluster) {
    for (period in c(0, 1)) {
      y <- 4.0 + cluster_effect + 1.0 * period
      if (is_treated == 1 && period == 1) y <- y + 0.3  # weak effect
      y <- y + rnorm(1, 0, 4.0)                          # heavy noise -> interior p
      rows[[i]] <- data.frame(cluster = c, treated = is_treated, post = period, y = y)
      i <- i + 1
    }
  }
}
df <- do.call(rbind, rows)
df$inter <- df$treated * df$post

data_path <- "benchmarks/data/wild_cluster_boot_test_data.csv"
write.csv(df[, c("cluster", "treated", "post", "y")], data_path, row.names = FALSE)

m <- feols(y ~ treated + post + inter, data = df, cluster = ~cluster)
coef_est <- as.numeric(coef(m)["inter"])
se_cr1 <- as.numeric(se(m)["inter"])  # CR1 clustered SE

run_bt <- function(ptype) {
  set.seed(1)
  bt <- boottest(m, param = "inter", clustid = ~cluster, B = 99999,
                 type = "rademacher", impose_null = TRUE,
                 p_val_type = ptype, conf_int = TRUE, sign_level = 0.05)
  list(p_val = as.numeric(bt$p_val),
       t_stat = as.numeric(bt$t_stat),
       conf_int = as.numeric(bt$conf_int),
       n_clusters = as.integer(bt$N_G))
}

golden <- list(
  n_clusters = G,
  coef = coef_est,
  se_cr1 = se_cr1,
  two_tailed = run_bt("two-tailed"),
  equal_tailed = run_bt("equal-tailed")
)

json_path <- "benchmarks/data/wild_cluster_boot_golden.json"
write(toJSON(golden, auto_unbox = TRUE, digits = 12, pretty = TRUE), json_path)
cat("Wrote", data_path, "and", json_path, "\n")
cat(sprintf("coef=%.6f se=%.6f two-tailed p=%.6f CI=[%.6f, %.6f]\n",
            coef_est, se_cr1, golden$two_tailed$p_val,
            golden$two_tailed$conf_int[1], golden$two_tailed$conf_int[2]))
