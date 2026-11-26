# External chained DiD reference environment

Build with `DOCKER_BUILDKIT=0 docker build -t diff-diff-cdid-reference:1.1.0.0 .`
from this directory. The explicit legacy-builder setting works on the inspected
host, whose Docker installation has no buildx component.

The base image is digest-pinned. R 4.5.2, cdid 1.1.0.0 at commit
1039120668cceabd8111cad82d32ab3927c2b94f, did 2.5.1 and DRDID 1.3.0 are
asserted during build and independently checked by the benchmark script.
`packages.json` records the full installed environment used for the comparison.
Transitive CRAN dependencies are resolved at build time; the recipe does not
promise a hermetic rebuild of every transitive package. The executed image ID
and this manifest's hash are recorded in the comparison output. Before fitting,
the comparator verifies the selected image's R/platform and every installed
package/version/build against the manifest; it refuses a changed inventory.
An intentional environment refresh therefore requires a separately reviewed
manifest update, rather than silently relabelling a rebuilt image.

Run `benchmarks/python/compare_chained_did.py` from the repository root. It
mounts the source read-only, writes inputs/outputs into its declared workspace
under the caller's UID, and never publishes an image or installs R packages
into the host. Its nonzero exit retains failing comparisons rather than
turning a mismatch into reference agreement.
The report also retains the asserted public callback weight probe from
`benchmarks/R/inspect_chained_did_weights.R`; that probe stops before estimation.
