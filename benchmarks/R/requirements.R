#!/usr/bin/env Rscript
# R Package Requirements for diff-diff Benchmarks
#
# Run this script to install all required R packages:
#   Rscript benchmarks/R/requirements.R

required_packages <- c(
  # Core DiD packages
  "did",           # Callaway-Sant'Anna (2021) staggered DiD
  "didimputation", # Borusyak, Jaravel & Spiess (2024) imputation DiD
  "HonestDiD",     # Rambachan & Roth (2023) sensitivity analysis
  "fixest",        # Fast TWFE and basic DiD
  "triplediff",    # Ortiz-Villavicencio & Sant'Anna (2025) triple difference
  "etwfe",         # McDermott (2023) Wooldridge ETWFE estimator (R-parity for Poisson + logit paths)
  "survey",        # Lumley (2004) complex survey analysis
  "estimatr",      # Blair et al. (2019) weighted robust / IV SE (HAD mass-point parity)
  "DIDHAD",        # de Chaisemartin et al. (2025) HAD estimator (HAD Phase 4 R-parity)
  "YatchewTest",   # Yatchew (1997) linearity test (HAD yatchew R-parity)
  "nprobust",      # Calonico-Cattaneo-Farrell local-linear (DIDHAD dependency)
  "Synth",         # Abadie-Diamond-Hainmueller (2010) synthetic control (SyntheticControl R-parity; ships data(basque))
  "qte",           # Callaway qte package (Athey-Imbens CiC + QDiD R-parity; ships data(lalonde))

  # Utilities
  "jsonlite",      # JSON output for Python interop
  "data.table"     # Fast data manipulation
)

# synthdid must be installed from GitHub
github_packages <- list(
  synthdid = "synth-inference/synthdid"
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing %s...", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
  } else {
    message(sprintf("%s is already installed.", pkg))
  }
}

# PR #392 R6 P3: pinned-version installer for upstream packages whose
# version is part of the parity contract. The HAD R-parity test
# (`tests/test_did_had_parity.py`) and the generator
# (`benchmarks/R/generate_did_had_golden.R`) hard-pin DIDHAD,
# YatchewTest, and nprobust to specific versions; without a
# version-aware installer here, a fresh R environment would silently
# install whatever CRAN currently serves and the generator's
# `stopifnot(packageVersion(...) == "X.Y.Z")` would abort.
install_pinned_version <- function(pkg, version) {
  if (requireNamespace(pkg, quietly = TRUE) &&
      as.character(packageVersion(pkg)) == version) {
    message(sprintf("%s is already at pinned version %s.", pkg, version))
    return(invisible(NULL))
  }
  message(sprintf("Installing %s == %s (pinned for HAD R-parity)...", pkg, version))
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "https://cloud.r-project.org/", quiet = TRUE)
  }
  remotes::install_version(
    pkg,
    version = version,
    repos = "https://cloud.r-project.org/",
    quiet = TRUE,
    upgrade = "never"
  )
}

# HAD R-parity (PR #392) version pins. Bump these in lockstep with
# the generator's `stopifnot(packageVersion(...) == "X.Y.Z")` and the
# parity test's `test_metadata_versions_match` when re-anchoring.
pinned_versions <- list(
  DIDHAD = "2.0.0",
  YatchewTest = "1.1.1",
  nprobust = "0.5.0",
  # CiC/QDiD R-parity (generate_qte_golden.R + tests/test_changes_in_changes_parity.py)
  qte = "1.3.1"
)

install_github_if_missing <- function(pkg, repo) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing %s from GitHub...", pkg))
    if (!requireNamespace("remotes", quietly = TRUE)) {
      install.packages("remotes", repos = "https://cloud.r-project.org/", quiet = TRUE)
    }
    remotes::install_github(repo, quiet = TRUE)
  } else {
    message(sprintf("%s is already installed.", pkg))
  }
}

# Install CRAN packages
message("Installing CRAN packages...")
lapply(required_packages, install_if_missing)

# Reinforce the HAD R-parity pinned versions AFTER the bulk install
# above (which may have installed any-CRAN-version of e.g. nprobust
# as a transitive dep). install_pinned_version is idempotent if the
# correct version is already installed.
message("\nEnforcing HAD R-parity version pins...")
for (pkg in names(pinned_versions)) {
  install_pinned_version(pkg, pinned_versions[[pkg]])
}

# Install GitHub packages
message("\nInstalling GitHub packages...")
for (pkg in names(github_packages)) {
  install_github_if_missing(pkg, github_packages[[pkg]])
}

# Verify installation
message("\nVerifying installation...")
all_packages <- c(required_packages, names(github_packages))
installed <- sapply(all_packages, requireNamespace, quietly = TRUE)

if (all(installed)) {
  message("\nAll packages installed successfully!")
} else {
  missing <- all_packages[!installed]
  stop(sprintf("Failed to install: %s", paste(missing, collapse = ", ")))
}

# Print versions
message("\nInstalled versions:")
for (pkg in all_packages) {
  version <- as.character(packageVersion(pkg))
  message(sprintf("  %s: %s", pkg, version))
}
