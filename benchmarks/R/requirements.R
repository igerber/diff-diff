#!/usr/bin/env Rscript
# R Package Requirements for diff-diff Benchmarks
#
# Run this script to install all required R packages:
#   Rscript benchmarks/R/requirements.R

required_packages <- c(
  # Core DiD packages
  "did",           # Callaway-Sant'Anna (2021) staggered DiD
  "HonestDiD",     # Rambachan & Roth (2023) sensitivity analysis
  "fixest",        # Fast TWFE and basic DiD

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
