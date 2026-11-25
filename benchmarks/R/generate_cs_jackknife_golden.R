#!/usr/bin/env Rscript
# Golden values for CallawaySantAnna(vcov_type="jackknife"): the cluster
# jackknife (CV3) of Karim, Nielsen, MacKinnon & Webb (2026, arXiv:2602.12043),
# equation (20), computed with R `did` on delete-one-cluster samples.
#
# Usage:   Rscript benchmarks/R/generate_cs_jackknife_golden.R
# Output:  benchmarks/data/cs_jackknife_golden.json
#
# The procedure is the one the paper's R package `didjack` implements
# (https://github.com/liu-yunhan/didjack, MIT): refit did::att_gt on each
# delete-one-cluster sample, re-aggregate with did::aggte, and take
#   CV3 = ((H - 1) / H) * sum_h (theta^(h) - theta)^2
# around the full-sample estimate, with a t(H - 1) reference. `didjack`
# reports the overall ATT only; the same formula is applied here element-wise
# to the dynamic and group aggregates, which is the library's documented
# extension. `didjack` itself is not on CRAN and is not required: only `did`
# point estimates enter, so no clustering option is passed to att_gt.
#
# The panel is embedded in the JSON so the Python test refits exactly it.
suppressMessages({
  library(did)
  library(jsonlite)
})

set.seed(20261125)

n_states <- 12L
units_per <- 3L
n_periods <- 6L
cohort_of_state <- c(3, 3, 3, 5, 5, 5, rep(0, 6))

rows <- list()
k <- 1L
for (s in seq_len(n_states)) {
  shock <- cumsum(rnorm(n_periods, 0, 0.4))
  g <- cohort_of_state[s]
  for (j in seq_len(units_per)) {
    u <- (s - 1L) * units_per + j
    ufe <- rnorm(1)
    for (t in seq_len(n_periods)) {
      eff <- if (g != 0 && t >= g) 0.8 + 0.2 * (t - g) else 0
      rows[[k]] <- data.frame(
        unit = u, state = sprintf("s%02d", s), t = t, g = g,
        y = ufe + 0.3 * t + shock[t] + eff + rnorm(1, 0, 0.5)
      )
      k <- k + 1L
    }
  }
}
df <- do.call(rbind, rows)
clusters <- unique(df$state)
H <- length(clusters)

estimates <- function(data, est_method, control_group, base_period) {
  out <- suppressWarnings(att_gt(
    yname = "y", tname = "t", idname = "unit", gname = "g", data = data,
    control_group = control_group, est_method = est_method,
    base_period = base_period, bstrap = FALSE, cband = FALSE
  ))
  simple <- aggte(out, type = "simple", na.rm = TRUE, bstrap = FALSE, cband = FALSE)
  dyn <- aggte(out, type = "dynamic", na.rm = TRUE, bstrap = FALSE, cband = FALSE)
  grp <- aggte(out, type = "group", na.rm = TRUE, bstrap = FALSE, cband = FALSE)
  list(
    overall = simple$overall.att,
    dyn_e = dyn$egt, dyn_att = dyn$att.egt,
    grp_g = grp$egt, grp_att = grp$att.egt
  )
}

cv3_se <- function(full, reps) sqrt((H - 1) / H * sum((reps - full)^2))

configurations <- list(
  reg_nevertreated_varying = list("reg", "nevertreated", "varying"),
  dr_nevertreated_varying = list("dr", "nevertreated", "varying"),
  reg_notyettreated_varying = list("reg", "notyettreated", "varying"),
  dr_nevertreated_universal = list("dr", "nevertreated", "universal")
)
library_control_group <- c(nevertreated = "never_treated", notyettreated = "not_yet_treated")

golden_cfg <- list()
for (name in names(configurations)) {
  cfg <- configurations[[name]]
  full <- estimates(df, cfg[[1]], cfg[[2]], cfg[[3]])
  reps <- lapply(clusters, function(cl) {
    estimates(df[df$state != cl, ], cfg[[1]], cfg[[2]], cfg[[3]])
  })
  for (r in reps) {
    stopifnot(identical(r$dyn_e, full$dyn_e), identical(r$grp_g, full$grp_g))
  }
  overall_loo <- vapply(reps, function(r) r$overall, numeric(1))
  se <- cv3_se(full$overall, overall_loo)
  t_stat <- full$overall / se
  crit <- qt(0.975, H - 1)
  dyn_loo <- do.call(rbind, lapply(reps, function(r) r$dyn_att))
  grp_loo <- do.call(rbind, lapply(reps, function(r) r$grp_att))
  dyn_se <- vapply(seq_along(full$dyn_e), function(j) cv3_se(full$dyn_att[j], dyn_loo[, j]), numeric(1))
  grp_se <- vapply(seq_along(full$grp_g), function(j) cv3_se(full$grp_att[j], grp_loo[, j]), numeric(1))
  # A universal base period makes one horizon a reference (estimate 0 in every
  # sample, hence a zero jackknife variance); the library reports NaN there.
  live <- dyn_se > 0
  golden_cfg[[name]] <- list(
    est_method = cfg[[1]],
    control_group = unname(library_control_group[cfg[[2]]]),
    base_period = cfg[[3]],
    df = H - 1,
    overall_att = full$overall,
    overall_att_loo = overall_loo,
    overall_se = se,
    overall_p = 2 * pt(-abs(t_stat), H - 1),
    overall_lci = full$overall - crit * se,
    overall_uci = full$overall + crit * se,
    event_times = full$dyn_e[live],
    event_study_att = full$dyn_att[live],
    event_study_se = dyn_se[live],
    group_labels = full$grp_g,
    group_att = full$grp_att,
    group_se = grp_se
  )
}

golden <- list(
  meta = list(
    did_version = as.character(packageVersion("did")),
    r_version = R.version.string,
    n_clusters = H,
    note = paste0(
      "Cluster jackknife (CV3, Karim et al. 2026 eq. 20) from did::att_gt + ",
      "aggte on delete-one-cluster samples; t(H-1) reference. Overall ATT as ",
      "in the didjack package; dynamic and group standard errors by the same ",
      "formula element-wise."
    )
  ),
  clusters = clusters,
  data = as.list(df),
  configurations = golden_cfg
)

out_path <- file.path("benchmarks", "data", "cs_jackknife_golden.json")
write_json(golden, out_path, digits = NA, auto_unbox = TRUE, pretty = TRUE)
cat("Wrote", out_path, "\n")
