#!/usr/bin/env Rscript
# External reference only; refuses unpinned environments and never installs packages.
# Usage: Rscript benchmarks/R/benchmark_chained_did.R input.csv output.json
# Input columns: id, t, g, y and optionally static x / cluster.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Expected input.csv and output.json")
if (getRversion() != "4.5.2") stop("Reference requires R 4.5.2")
versions <- c(cdid = "1.1.0.0", did = "2.5.1", DRDID = "1.3.0")
for (package in names(versions)) {
  if (!requireNamespace(package, quietly = TRUE)) stop(paste("Missing reference package", package))
  if (as.character(packageVersion(package)) != versions[[package]]) {
    stop(paste("Unmatched reference version for", package))
  }
}
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Missing jsonlite")
sha <- packageDescription("cdid")[["RemoteSha"]]
if (is.null(sha) || sha != "1039120668cceabd8111cad82d32ab3927c2b94f") {
  stop("cdid must record the pinned GitHub RemoteSha")
}
data <- read.csv(args[[1L]], na.strings = c("", "NA"))
stopifnot(all(c("id", "t", "g", "y") %in% names(data)))
data <- data[order(data$id, data$t), ]
frame_columns <- intersect(c("id", "g", "x", "cluster"), names(data))
frame <- data[!duplicated(data$id), frame_columns, drop = FALSE]
fit <- cdid::att_gt(
  yname = "y", tname = "t", idname = "id", gname = "g", data = data, frame = frame,
  xformla = if ("x" %in% names(data)) ~ x else ~ 1,
  control_group = "nevertreated", anticipation = 0, est_method = "ipw",
  base_period = "universal", mode = "chained", selection = "none",
  gmm_weighting = "identity", pt_assumption = "post",
  bstrap = FALSE, cband = FALSE,
  clustervars = if ("cluster" %in% names(data)) "cluster" else NULL
)
simple <- cdid::aggte(fit, type = "simple", bstrap = FALSE, cband = FALSE)
dynamic <- cdid::aggte(fit, type = "dynamic", bstrap = FALSE, cband = FALSE)
event_if <- dynamic$inf.function$dynamic.inf.func.e
stopifnot(is.matrix(event_if), nrow(event_if) == fit$n)
if ("cluster" %in% names(data)) {
  positions <- match(rownames(event_if), as.character(frame$id))
  stopifnot(!anyNA(positions))
  event_if <- rowsum(event_if, frame$cluster[positions], reorder = FALSE)
}
event_vcov <- crossprod(event_if) / fit$n^2
finite_se <- is.finite(dynamic$se.egt)
stopifnot(all(abs(sqrt(diag(event_vcov))[finite_se] - dynamic$se.egt[finite_se]) < 1e-8))
calendar <- sort(unique(data$t))
groups <- sort(unique(data$g[data$g > 0]))
references <- calendar[match(groups, calendar) - 1L]
event_reference <- vapply(dynamic$egt, function(e) {
  included <- (fit$t - fit$group) == e
  any(included) && all(fit$t[included] == references[match(fit$group[included], groups)])
}, logical(1))
answer <- list(
  metadata = list(R = as.character(getRversion()), packages = as.list(versions),
                  cdid_sha = sha, n = fit$n, covariance_scale = "estimator",
                  cluster_correction = "none"),
  cells = data.frame(group = fit$group, time = fit$t, att = fit$att,
                     available = is.finite(fit$att)),
  vcov = fit$V_analytical / fit$n,
  overall = list(att = simple$overall.att, se = simple$overall.se),
  events = list(event_time = dynamic$egt, att = dynamic$att.egt, se = dynamic$se.egt,
                is_reference = event_reference, vcov = event_vcov)
)
jsonlite::write_json(answer, args[[2L]], auto_unbox = TRUE, pretty = TRUE,
                     na = "null", digits = 17)
