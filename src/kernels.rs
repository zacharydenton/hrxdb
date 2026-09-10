//! Kernel source is authored in Loom; Rust supplies only specialization values.
use crate::ScanConfig;
use hrx::loom::Specialization;
pub const SCAN: &str = include_str!("../kernels/scan_family.loom");
pub const SELECT: &str = include_str!("../kernels/select_family.loom");
pub fn scan_spec(n: usize, d: usize, c: ScanConfig, control: bool) -> Specialization {
    let mut spec = Specialization::new("scan");
    for (key, value) in [
        ("db.scan.count", n),
        ("db.scan.dimensions", d),
        ("db.scan.threads", c.threads),
        ("db.scan.rows", c.rows_per_wave),
        ("db.scan.width", c.load_width),
        ("db.scan.mode", usize::from(control)),
    ] {
        spec.config.insert(key.into(), value.to_string());
    }
    spec.report = true;
    spec
}
pub fn select_spec(first: bool) -> Specialization {
    let mut spec = Specialization::new("select");
    spec.config
        .insert("db.select.first".into(), usize::from(first).to_string());
    spec.report = true;
    spec
}
