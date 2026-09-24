use anyhow::{Context, Result};
use std::path::Path;

pub struct PdfPage {
    pub page: usize,
    pub text: String,
}

/// Извлекает текст PDF постранично (1-indexed).
pub fn extract_pdf_pages(path: &Path) -> Result<Vec<PdfPage>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("cannot read pdf: {}", path.display()))?;
    let pages: Vec<String> =
        pdf_extract::extract_text_from_mem_by_pages(&bytes).with_context(|| {
            format!("pdf text extraction failed (encrypted/unsupported?): {}", path.display())
        })?;
    Ok(pages
        .into_iter()
        .enumerate()
        .map(|(i, text)| PdfPage { page: i + 1, text })
        .collect())
}