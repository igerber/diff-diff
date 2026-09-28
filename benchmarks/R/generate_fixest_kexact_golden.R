# Golden: fixest exact-vs-default FE counting on a hierarchical two-way design.
#
# absorb = [state, state_year] with state_year nested in state splits the
# bipartite level graph into one component per state (C = 6), so the absorbed
# dummy-space rank is 29 beyond the intercept -- not the naive
# sum(levels - 1) = 34. fixest's default ssc(K.exact = FALSE) uses the naive
# count (df.K = 36 here); ssc(K.exact = TRUE) computes the exact rank
# (df.K = 31). diff-diff's component-aware absorbed_fe_rank matches the EXACT
# side at machine precision (a documented deviation from the R *default*).
#
# Regenerate: Rscript benchmarks/R/generate_fixest_kexact_golden.R
suppressMessages(library(fixest))
suppressMessages(library(jsonlite))

set.seed(7)
d <- expand.grid(s = 0:5, y = 0:4, r = 1:4)
d$state <- d$s
d$state_year <- d$s * 100 + d$y
d$x <- rnorm(nrow(d))
d$out <- rnorm(nrow(d)) + 0.5 * d$x + 0.3 * d$s

m_default <- feols(out ~ x | state + state_year, data = d, vcov = "iid")
m_exact <- feols(out ~ x | state + state_year, data = d, vcov = "iid",
                 ssc = ssc(K.exact = TRUE))

golden <- list(
  meta = list(
    generator = "benchmarks/R/generate_fixest_kexact_golden.R",
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    fixest_version = as.character(packageVersion("fixest")),
    description = paste(
      "Hierarchical two-way FE (state_year nested in state, C=6):",
      "fixest default ssc(K.exact=FALSE) vs exact FE-rank counting.",
      "diff-diff absorbed_fe_rank matches the K.exact=TRUE side."
    )
  ),
  data = list(
    state = d$state,
    state_year = d$state_year,
    x = d$x,
    out = d$out
  ),
  n_obs = nrow(d),
  coef = unname(coef(m_default)[["x"]]),
  iid_default = list(
    se = unname(se(m_default)[["x"]]),
    df_k = degrees_freedom(m_default, "k")
  ),
  iid_k_exact = list(
    se = unname(se(m_exact)[["x"]]),
    df_k = degrees_freedom(m_exact, "k")
  )
)

path <- "benchmarks/data/fixest_kexact_golden.json"
write_json(golden, path, digits = NA, auto_unbox = TRUE, pretty = TRUE)
cat("wrote", path, "\n")
cat(sprintf("coef=%.15f default_se=%.15f exact_se=%.15f\n",
            golden$coef, golden$iid_default$se, golden$iid_k_exact$se))
