#!/usr/bin/env Rscript
# Golden values for CallawaySantAnna(allow_unbalanced_panel=True) parity vs
# R did::att_gt(allow_unbalanced_panel=TRUE) + aggte(type="dynamic").
#
# Usage:   Rscript benchmarks/R/generate_cs_unbalanced_golden.R
# Output:  benchmarks/data/cs_unbalanced_golden.json
#
# R's allow_unbalanced_panel=TRUE sets panel=FALSE and runs the repeated-cross-
# section levels estimator (DRDID::reg_did_rc) on the pooled observations.
# diff-diff matches the ATT bit-for-bit; the analytical SE matches up to the
# CR1 G/(G-1) finite-sample factor diff-diff's cluster path applies (see the
# test + REGISTRY note). The panel is deliberately UNBALANCED (attrition in
# cohort 3 at t>=4) so a cell's valid-unit count < its cohort mass, exercising
# the obs-vs-unit pg weighting and the RC-on-panel estimator.
suppressMessages({
  library(did)
  library(jsonlite)
})

set.seed(20260706)

make_panel <- function() {
  rows <- list()
  k <- 1L
  add_unit <- function(u, g, drop_late) {
    ufe <- rnorm(1)
    for (t in 1:5) {
      if (drop_late && t >= 4 && runif(1) < 0.40) next
      post <- if (g != 0 && t >= g) 1 else 0
      eff <- if (g == 3 && post) 1.0 * (t - g + 1) else if (g == 4 && post) 2.0 * (t - g + 1) else 0
      rows[[k]] <<- data.frame(unit = u, period = t, g = g, y = ufe + 0.3 * t + eff + rnorm(1, 0, 0.5))
      k <<- k + 1L
    }
  }
  u <- 0L
  for (i in 1:50) { add_unit(u, 3, TRUE); u <- u + 1L }
  for (i in 1:50) { add_unit(u, 4, FALSE); u <- u + 1L }
  for (i in 1:100) { add_unit(u, 0, FALSE); u <- u + 1L }
  do.call(rbind, rows)
}

df <- make_panel()

out <- att_gt(
  yname = "y", tname = "period", idname = "unit", gname = "g", data = df,
  control_group = "nevertreated", est_method = "reg",
  allow_unbalanced_panel = TRUE, bstrap = FALSE, cband = FALSE
)
agg <- aggte(out, type = "dynamic", na.rm = TRUE, bstrap = FALSE, cband = FALSE)
agg_simple <- aggte(out, type = "simple", na.rm = TRUE, bstrap = FALSE, cband = FALSE)
agg_group <- aggte(out, type = "group", na.rm = TRUE, bstrap = FALSE, cband = FALSE)

golden <- list(
  meta = list(
    did_version = as.character(packageVersion("did")),
    n_units = length(unique(df$unit)),
    note = paste0(
      "allow_unbalanced_panel=TRUE (panel=FALSE -> DRDID::reg_did_rc on pooled ",
      "obs). ATT bit-exact vs diff-diff; SE parity up to the CR1 G/(G-1) factor."
    )
  ),
  data = list(
    unit = as.numeric(df$unit),
    period = as.numeric(df$period),
    first_treat = as.numeric(df$g),
    outcome = as.numeric(df$y)
  ),
  cells = list(
    g = as.numeric(out$group),
    t = as.numeric(out$t),
    att = as.numeric(out$att),
    se = as.numeric(out$se)
  ),
  event_study = list(
    egt = as.numeric(agg$egt),
    att = as.numeric(agg$att.egt),
    se = as.numeric(agg$se.egt),
    overall_att = as.numeric(agg$overall.att),
    overall_se = as.numeric(agg$overall.se)
  ),
  simple = list(
    overall_att = as.numeric(agg_simple$overall.att),
    overall_se = as.numeric(agg_simple$overall.se)
  ),
  group = list(
    egt = as.numeric(agg_group$egt),
    att = as.numeric(agg_group$att.egt),
    se = as.numeric(agg_group$se.egt),
    overall_att = as.numeric(agg_group$overall.att),
    overall_se = as.numeric(agg_group$overall.se)
  )
)

out_path <- file.path("benchmarks", "data", "cs_unbalanced_golden.json")
writeLines(toJSON(golden, auto_unbox = TRUE, digits = 16, pretty = TRUE), out_path)
cat("Wrote", out_path, "\n")
