pub mod pdf;
pub mod xlsx;

use anyhow::Context;

pub fn is_supported(path: &std::path::Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()),
        Some(ref e) if e == "pdf" || e == "xlsx" || e == "xlsm"
    )
}

pub fn supported_files(dir: &std::path::Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("cannot read input directory: {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && is_supported(&path) {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}