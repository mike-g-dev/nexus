use std::path::PathBuf;

fn main() {
    let out = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let fixtures =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../nexus-fix-codegen/tests/fixtures");
    let alpha = fixtures.join("venue_alpha.xml");
    let beta = fixtures.join("venue_beta.xml");
    println!("cargo:rerun-if-changed={}", alpha.display());
    println!("cargo:rerun-if-changed={}", beta.display());
    nexus_fix_codegen::generate()
        .dictionary(&alpha)
        .dictionary(&beta)
        .out_dir(&out)
        .rustfmt(false)
        .run()
        .expect("codegen failed");
}
