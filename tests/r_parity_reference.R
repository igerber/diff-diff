args <- commandArgs(trailingOnly = TRUE)
mode <- args[[1]]
input <- args[[2]]
output <- args[[3]]
data <- read.csv(input, check.names = FALSE)

if (mode == "twfeweights") {
  suppressPackageStartupMessages({
    library(did)
    library(twfeweights)
  })
  result <- suppressWarnings(did::att_gt(
    yname = "Y", tname = "period", idname = "id", gname = "G",
    data = data, control_group = "nevertreated", base_period = "universal",
    bstrap = FALSE
  ))
  out <- twfeweights::twfe_weights(result, keep_untreated = TRUE)$weights_df
  write.csv(out, output, row.names = FALSE)
} else if (mode == "ptetools") {
  suppressPackageStartupMessages(library(ptetools))
  subset <- ptetools::two_by_two_subset(data, g = 2, tp = 2)
  result <- ptetools::did_attgt(subset$gt_data)
  write.csv(data.frame(att = result$attgt), output, row.names = FALSE)
} else if (mode == "badcontrols") {
  suppressPackageStartupMessages(library(badcontrols))
  result <- badcontrols::didbc(
    yname = "Y", gname = "G", tname = "period", idname = "id", data = data,
    bad_control_formula = ~X, xformula = ~1, est_method = "imputation",
    bstrap = FALSE, cband = FALSE
  )
  extracted <- badcontrols::extract_att(result)
  write.csv(data.frame(att = extracted$att, se = extracted$se), output, row.names = FALSE)
} else {
  stop("unknown parity mode")
}
