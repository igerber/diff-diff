FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

ENV MAKEFLAGS=-j2
RUN Rscript -e 'options(repos=c(CRAN="https://cloud.r-project.org")); install.packages(c("remotes", "jsonlite")); remotes::install_version("DRDID", version="1.3.0", dependencies=NA, upgrade="never"); remotes::install_version("did", version="2.5.1", dependencies=NA, upgrade="never"); remotes::install_github("dvbn/cdid@1039120668cceabd8111cad82d32ab3927c2b94f", dependencies=FALSE, upgrade="never", build_vignettes=FALSE); stopifnot(getRversion()=="4.5.2", packageVersion("cdid")=="1.1.0.0", packageVersion("did")=="2.5.1", packageVersion("DRDID")=="1.3.0")'
WORKDIR /work
