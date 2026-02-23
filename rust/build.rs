/// Build script for diff_diff_rust.
///
/// When the `openblas` feature is enabled, links against the system OpenBLAS
/// library directly. This avoids the `openblas-src` -> `openblas-build` ->
/// `ureq` -> `native-tls` dependency chain, which has Rust compiler
/// compatibility issues. Requires `libopenblas-dev` (Ubuntu) or
/// `openblas-devel` (CentOS/manylinux) to be installed.
fn main() {
    if std::env::var("CARGO_FEATURE_OPENBLAS").is_ok() {
        println!("cargo:rustc-link-lib=openblas");
    }
}
