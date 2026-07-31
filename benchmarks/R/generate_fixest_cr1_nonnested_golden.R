# Golden: fixest clustered CR1 with a NON-NESTED absorbed-FE rank term.
#
# The committed clustered fixest arms (fixest_did_twfe_golden.json) exercise
# only connected two-way panels, where the non-nested rank term is the trivial
# delta = T - 1. What no golden covered is the RANK of the non-nested absorbed
# FE on an irregular design -- the exact quantity the K_reference convention
# adds to the clustered CR1 k (variance-conventions.md D2).
#
# DGP: a DISCONNECTED two-way [unit, time] panel -- two blocks with disjoint
# period ranges (C = 2 bipartite components) -- plus a fully-crossed connected
# control panel. Three arms:
#
# 1. disconnected$crossed_cluster (THE parity anchor): clustered by
#    c5 = (unit + time) %% 5, which crosses both dims, so NOTHING is nested
#    and the K_reference increment is the exact full span rank
#    U + T - C = 28 (K = 29 with x). fixest ssc(K.fixef = "full",
#    K.exact = TRUE) computes exactly that (df.K = 29, agreeing with iid
#    K.exact and with reghdfe's df_a = 28 + x); the default approximate
#    count gives df.K = 30, and K.exact under the DEFAULT
#    K.fixef = "nested" gives df.K = 28 -- on the clustered path fixest's
#    nested handling removes 1 df even when nothing is nested (recorded as
#    k_exact_nested for documentation). All three are recorded; diff-diff
#    must match the full+K.exact side at 1e-12 and differ from the other
#    two (the same deviation-from-R-default as the kexact sibling golden).
#    This arm also pins the zero-nested-dim max(..., 1) floor end-to-end.
#
# 2. disconnected$unit_cluster (DOCUMENTATION arm, no parity target):
#    clustered by unit (nested, dropped; exact remainder rank of time given
#    unit is 28 - 20 = 8, so K_reference = x + constant + 8 = 10, denominator
#    n - 10). NO fixest ssc configuration produces that composition:
#    the default nested+approximate count gives df.K = 11 (time counted
#    T - 1 = 9), and ssc(K.exact = TRUE) composes INCOHERENTLY with the
#    nested drop (df.K = 28 -- it removes only 1 df for the fully-nested
#    unit FE). Both are recorded so the consuming test can pin that the
#    library matches NEITHER: its exact-remainder K = 10 sits one df below
#    fixest's default (Stata reghdfe lands on the same 11 via its own
#    approximate remainder -- see reghdfe_kref_golden.json). NO external
#    reference implements nested-drop + exact-remainder; the library's
#    convention is the consistent extension of the exact-rank principle,
#    and the test pins the deviation exactly:
#    se_default / se_library == sqrt((n - 10) / (n - 11)).
#
# 3. connected (control): fully-crossed two-way panel clustered by unit,
#    default ssc -- at C = 1 with independent dims the two conventions
#    coincide (rank term T - 1), pinning the regular case.
#
# Regenerate: Rscript benchmarks/R/generate_fixest_cr1_nonnested_golden.R
suppressMessages(library(fixest))
suppressMessages(library(jsonlite))

set.seed(11)

# --- Disconnected two-way panel (C = 2) --------------------------------------
d1 <- rbind(
  expand.grid(unit = 0:9, time = 0:4),
  expand.grid(unit = 10:19, time = 5:9)
)
d1$x <- rnorm(nrow(d1))
d1$out <- rnorm(nrow(d1)) + 0.5 * d1$x + 0.2 * (d1$unit %% 3) + 0.1 * d1$time
d1$c5 <- (d1$unit + d1$time) %% 5

m1_unit_default <- feols(out ~ x | unit + time, data = d1, cluster = ~unit)
m1_unit_kexact <- feols(out ~ x | unit + time, data = d1, cluster = ~unit,
                        ssc = ssc(K.exact = TRUE))
m1_cross_kexact <- feols(out ~ x | unit + time, data = d1, cluster = ~c5,
                         ssc = ssc(K.fixef = "full", K.exact = TRUE))
m1_cross_kexact_nested <- feols(out ~ x | unit + time, data = d1, cluster = ~c5,
                                ssc = ssc(K.exact = TRUE))
m1_cross_default <- feols(out ~ x | unit + time, data = d1, cluster = ~c5)

# --- Connected control (C = 1), clustered by unit, default ssc ---------------
d2 <- expand.grid(unit = 0:19, time = 0:9)
d2$x <- rnorm(nrow(d2))
d2$out <- rnorm(nrow(d2)) + 0.5 * d2$x + 0.2 * (d2$unit %% 3) + 0.1 * d2$time

m2 <- feols(out ~ x | unit + time, data = d2, cluster = ~unit)

golden <- list(
  meta = list(
    generator = "benchmarks/R/generate_fixest_cr1_nonnested_golden.R",
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    fixest_version = as.character(packageVersion("fixest")),
    description = paste(
      "Clustered CR1 with a non-nested absorbed-FE RANK term on a",
      "disconnected two-way panel (C=2). crossed_cluster: nothing nested,",
      "K.exact=TRUE is the parity anchor (exact span rank 28, df.K=29) vs",
      "the default approximate count (df.K=30). unit_cluster: documentation",
      "arm -- no fixest ssc reproduces nested-drop + exact-remainder",
      "(library K=10 vs default df.K=11; reghdfe agrees with fixest's 11",
      "via its own approximate remainder, see reghdfe_kref_golden.json;",
      "the library deviation is pinned as se ratio sqrt((n-10)/(n-11))).",
      "connected: control arm where the conventions agree."
    )
  ),
  disconnected = list(
    data = list(
      unit = d1$unit,
      time = d1$time,
      x = d1$x,
      out = d1$out,
      c5 = d1$c5
    ),
    n_obs = nrow(d1),
    unit_cluster = list(
      n_clusters = length(unique(d1$unit)),
      coef = unname(coef(m1_unit_default)[["x"]]),
      default = list(
        se = unname(se(m1_unit_default)[["x"]]),
        df_k = degrees_freedom(m1_unit_default, "k")
      ),
      nested_k_exact = list(
        se = unname(se(m1_unit_kexact)[["x"]]),
        df_k = degrees_freedom(m1_unit_kexact, "k")
      )
    ),
    crossed_cluster = list(
      n_clusters = length(unique(d1$c5)),
      coef = unname(coef(m1_cross_kexact)[["x"]]),
      k_exact = list(
        se = unname(se(m1_cross_kexact)[["x"]]),
        df_k = degrees_freedom(m1_cross_kexact, "k")
      ),
      k_exact_nested = list(
        se = unname(se(m1_cross_kexact_nested)[["x"]]),
        df_k = degrees_freedom(m1_cross_kexact_nested, "k")
      ),
      default = list(
        se = unname(se(m1_cross_default)[["x"]]),
        df_k = degrees_freedom(m1_cross_default, "k")
      )
    )
  ),
  connected = list(
    data = list(
      unit = d2$unit,
      time = d2$time,
      x = d2$x,
      out = d2$out
    ),
    n_obs = nrow(d2),
    n_clusters = length(unique(d2$unit)),
    coef = unname(coef(m2)[["x"]]),
    cluster_default = list(
      se = unname(se(m2)[["x"]]),
      df_k = degrees_freedom(m2, "k")
    )
  )
)

path <- "benchmarks/data/fixest_cr1_nonnested_golden.json"
write_json(golden, path, digits = NA, auto_unbox = TRUE, pretty = TRUE)
cat("wrote", path, "\n")
dd <- golden$disconnected
cat(sprintf("unit_cluster:    coef=%.15f default df.K=%d se=%.15f | nested+K.exact df.K=%d se=%.15f\n",
            dd$unit_cluster$coef,
            dd$unit_cluster$default$df_k, dd$unit_cluster$default$se,
            dd$unit_cluster$nested_k_exact$df_k, dd$unit_cluster$nested_k_exact$se))
cat(sprintf("crossed_cluster: coef=%.15f k_exact df.K=%d se=%.15f | default df.K=%d se=%.15f\n",
            dd$crossed_cluster$coef,
            dd$crossed_cluster$k_exact$df_k, dd$crossed_cluster$k_exact$se,
            dd$crossed_cluster$default$df_k, dd$crossed_cluster$default$se))
cat(sprintf("connected:       coef=%.15f df.K=%d se=%.15f\n",
            golden$connected$coef,
            golden$connected$cluster_default$df_k, golden$connected$cluster_default$se))
