use crate::clean::CleanedPage;
use crate::config::ChunkingConfig;
use crate::extract::xlsx::SheetRecord;
use anyhow::Result;
use regex::Regex;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKind {
    Pdf,
    Xlsx,
}

#[derive(Debug, Clone)]
pub struct ChunkRecord {
    pub source: String,
    pub kind: ChunkKind,
    pub page: Option<usize>,
    pub section: Option<String>,
    pub sheet: Option<String>,
    pub row: Option<usize>,
    pub text: String,
}

fn est_chars_from_tokens(tokens: usize) -> usize {
    tokens * 4
}

/// Ведущий номер раздела: `6`, `6.3`, `6.3.5.3`.
fn section_number(line: &str) -> Option<String> {
    let mut num = String::new();
    for ch in line.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            num.push(ch);
        } else {
            break;
        }
    }
    if num.is_empty() || !num.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    Some(num.trim_end_matches('.').to_string())
}

fn heading_gap_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\s{2,}").expect("valid regex"))
}

/// Является ли строка заголовком раздела.
///
/// После очистки табличные строки размечены через `|`, а строки оглавления
/// удалены, поэтому дополнительные проверки здесь — страховка для режима
/// «без очистки»: раньше на них приходилось ~73% ложных заголовков.
fn is_section_heading(re: &Regex, line: &str) -> bool {
    if line.chars().count() >= 200 {
        return false;
    }
    if line.contains('|') {
        return false;
    }
    if line.contains("...") || line.contains(". . .") {
        return false;
    }
    // Строка таблицы, схлопнутая без разметки: несколько разрывов из 2+ пробелов.
    // У настоящего заголовка такой разрыв ровно один — между номером и названием.
    if heading_gap_re().find_iter(line).count() >= 2 {
        return false;
    }
    if !re.is_match(line) {
        return false;
    }
    let title = match section_number(line) {
        Some(n) => &line[n.len()..],
        None => return false,
    };
    let words = title
        .split_whitespace()
        .filter(|w| w.chars().filter(|c| c.is_alphabetic()).count() >= 2)
        .count();
    words >= 1
}

fn build_text(parts: &[String], section: &Option<String>, path: &Option<String>) -> String {
    let mut head: Vec<String> = Vec::new();
    match (section, path) {
        (Some(_), Some(p)) => head.push(format!("[Section] {p}")),
        (Some(s), None) => head.push(format!("[Section] {s}")),
        _ => {}
    }
    if head.is_empty() {
        return parts.join("\n");
    }
    head.push(parts.join("\n"));
    head.join("\n")
}

/// Иерархический путь раздела: `6.3.5 Laboratory Test Results > 6.3.5.3 Cell Phenotype Findings (CP)`.
fn section_path(
    number: &str,
    titles: &HashMap<String, String>,
    leaf: &str,
) -> Option<String> {
    let comps: Vec<&str> = number.split('.').collect();
    if comps.len() < 2 {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    for i in 1..comps.len() {
        let prefix = comps[..i].join(".");
        if let Some(t) = titles.get(&prefix) {
            parts.push(t.clone());
        }
    }
    if parts.is_empty() {
        return None;
    }
    parts.push(leaf.to_string());
    Some(parts.join(" > "))
}

struct FlushCtx<'a> {
    source: &'a str,
    overlap_chars: usize,
}

fn flush_buffer(
    buf: &mut Vec<String>,
    section: &Option<String>,
    path: &Option<String>,
    page: usize,
    ctx: &FlushCtx<'_>,
    out: &mut Vec<ChunkRecord>,
) {
    if buf.is_empty() {
        return;
    }
    let text = build_text(buf, section, path);
    let joined = buf.join("\n");
    let keep: String = {
        let rev: Vec<char> = joined.chars().rev().take(ctx.overlap_chars).collect();
        rev.into_iter().rev().collect()
    };
    out.push(ChunkRecord {
        source: ctx.source.to_string(),
        kind: ChunkKind::Pdf,
        page: Some(page),
        section: section.clone(),
        sheet: None,
        row: None,
        text,
    });
    buf.clear();
    if !keep.trim().is_empty() {
        buf.push(keep);
    }
}

pub fn chunk_pdf(pages: &[CleanedPage], source: &str, cfg: &ChunkingConfig) -> Result<Vec<ChunkRecord>> {
    let section_re = Regex::new(r"^\d{1,3}(?:\.\d{1,3}){0,4}\s+[A-Za-z].*$")?;
    let target_chars = est_chars_from_tokens(cfg.target_tokens).min(cfg.max_chunk_chars);
    let overlap_chars = est_chars_from_tokens(cfg.overlap_tokens);
    let ctx = FlushCtx { source, overlap_chars };

    let mut out: Vec<ChunkRecord> = Vec::new();
    let mut cur_section: Option<String> = None;
    let mut cur_path: Option<String> = None;
    let mut titles: HashMap<String, String> = HashMap::new();
    let mut buf: Vec<String> = Vec::new();
    let mut chars = 0usize;
    let mut buf_page = pages.first().map(|p| p.page).unwrap_or(0);

    for page in pages {
        for line in page.text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line.chars().count() <= 2 {
                continue;
            }
            if is_section_heading(&section_re, line) {
                flush_buffer(&mut buf, &cur_section, &cur_path, buf_page, &ctx, &mut out);
                chars = 0;
                cur_section = Some(line.to_string());
                let num = section_number(line).unwrap_or_default();
                cur_path = section_path(&num, &titles, line);
                titles.insert(num, line.to_string());
                buf_page = page.page;
                continue;
            }
            if buf.is_empty() {
                buf_page = page.page;
            }
            buf.push(line.to_string());
            chars += line.chars().count();
            if chars >= target_chars {
                flush_buffer(&mut buf, &cur_section, &cur_path, buf_page, &ctx, &mut out);
                chars = buf.iter().map(|s| s.chars().count()).sum();
            }
        }
    }
    flush_buffer(
        &mut buf,
        &cur_section,
        &cur_path,
        buf_page,
        &ctx,
        &mut out,
    );

    Ok(merge_small_chunks(out, cfg))
}

/// `None`/`None` считается совместимым; иначе второй раздел должен быть
/// тем же самым или вложенным в первый.
fn sections_related(a: &Option<String>, b: &Option<String>) -> bool {
    match (a, b) {
        (_, None) => true,
        (None, _) => true,
        (Some(x), Some(y)) => {
            if x == y {
                return true;
            }
            match (section_number(x), section_number(y)) {
                (Some(nx), Some(ny)) => ny.starts_with(&format!("{nx}.")),
                _ => false,
            }
        }
    }
}

/// Склеивает слишком короткие чанки с последующими — в пределах одной
/// страницы, одного источника и одного раздела (или его подраздела).
/// Страница сохраняется от первого чанка, поэтому локаторы для оценки не «плывут».
fn merge_small_chunks(chunks: Vec<ChunkRecord>, cfg: &ChunkingConfig) -> Vec<ChunkRecord> {
    if cfg.min_chunk_chars == 0 {
        return chunks;
    }
    let mut out: Vec<ChunkRecord> = Vec::with_capacity(chunks.len());
    let mut merged_pairs = 0usize;
    for c in chunks {
        let can_merge = match out.last() {
            Some(last) => {
                last.text.chars().count() < cfg.min_chunk_chars
                    && last.page == c.page
                    && last.source == c.source
                    && last.kind == c.kind
                    && sections_related(&last.section, &c.section)
                    && last.text.chars().count() + c.text.chars().count() <= cfg.max_chunk_chars
            }
            None => false,
        };
        if can_merge {
            let last = out.last_mut().expect("checked above");
            last.text.push('\n');
            last.text.push_str(&c.text);
            merged_pairs += 1;
        } else {
            out.push(c);
        }
    }
    if merged_pairs > 0 {
        eprintln!("[chunk] merged {merged_pairs} small chunk(s) into neighbours");
    }
    out
}

pub fn chunk_xlsx(records: &[SheetRecord], source: &str) -> Vec<ChunkRecord> {
    let mut out = Vec::new();
    for rec in records {
        let text = {
            let mut prefix = format!("[Sheet] {}\n", rec.sheet);
            for (header, value) in &rec.cells {
                prefix.push_str(&format!("{header}: {value}\n"));
            }
            prefix
        };
        if text.trim().is_empty() {
            continue;
        }
        out.push(ChunkRecord {
            source: source.to_string(),
            kind: ChunkKind::Xlsx,
            page: None,
            section: None,
            sheet: Some(rec.sheet.clone()),
            row: Some(rec.row),
            text,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cp(page: usize, text: &str) -> CleanedPage {
        CleanedPage {
            page,
            text: text.to_string(),
        }
    }

    #[test]
    fn rejects_table_rows_and_toc_as_sections() {
        let re = Regex::new(r"^\d{1,3}(?:\.\d{1,3}){0,4}\s+[A-Za-z].*$").unwrap();
        // настоящие заголовки
        assert!(is_section_heading(&re, "6.3.5 Laboratory Test Results"));
        assert!(is_section_heading(&re, "1 INTRODUCTION"));
        // строка таблицы после очистки размечена пайпами
        assert!(!is_section_heading(&re, "1 | Role | CDISC Notes | Core"));
        assert!(!is_section_heading(&re, "234 | CP | 0001"));
        // строка оглавления
        assert!(!is_section_heading(&re, "1.1 PURPOSE .............. 5"));
        // табличная строка без разметки (режим «без очистки»)
        assert!(!is_section_heading(&re, "1  Role  CDISC Notes  Core"));
    }

    #[test]
    fn builds_hierarchical_section_path() {
        let mut titles = HashMap::new();
        titles.insert("6".to_string(), "6 Laboratory Test Results".to_string());
        titles.insert("6.3".to_string(), "6.3 Findings".to_string());
        titles.insert(
            "6.3.5".to_string(),
            "6.3.5 Laboratory Test Results".to_string(),
        );
        let p = section_path(
            "6.3.5.3",
            &titles,
            "6.3.5.3 Cell Phenotype Findings (CP)",
        )
        .unwrap();
        assert_eq!(
            p,
            "6 Laboratory Test Results > 6.3 Findings > 6.3.5 Laboratory Test Results > 6.3.5.3 Cell Phenotype Findings (CP)"
        );
    }

    #[test]
    fn merges_small_chunks_on_same_page() {
        let cfg = ChunkingConfig {
            target_tokens: 20, // 80 символов — заведомо меньше min_chunk_chars
            overlap_tokens: 0,
            max_chunk_chars: 4000,
            min_chunk_chars: 600,
        };
        let pages = vec![cp(
            7,
            "1 First Section\nshort text\n1.1 Sub Section\nmore text here",
        )];
        let chunks = chunk_pdf(&pages, "doc.pdf", &cfg).unwrap();
        assert_eq!(chunks.len(), 1, "{chunks:#?}");
        assert_eq!(chunks[0].page, Some(7));
        assert_eq!(chunks[0].section.as_deref(), Some("1 First Section"));
    }

    #[test]
    fn does_not_merge_across_pages() {
        let cfg = ChunkingConfig {
            target_tokens: 20,
            overlap_tokens: 0,
            max_chunk_chars: 4000,
            min_chunk_chars: 600,
        };
        let pages = vec![
            cp(1, "1 First Section\nshort text"),
            cp(2, "1.1 Sub Section\nmore text here"),
        ];
        let chunks = chunk_pdf(&pages, "doc.pdf", &cfg).unwrap();
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn keeps_page_of_first_line_in_buffer() {
        let cfg = ChunkingConfig {
            target_tokens: 900,
            overlap_tokens: 0,
            max_chunk_chars: 4000,
            min_chunk_chars: 0,
        };
        let pages = vec![cp(1, "1 Section\nbody text"), cp(2, "continues here")];
        let chunks = chunk_pdf(&pages, "doc.pdf", &cfg).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].page, Some(1));
    }
}
