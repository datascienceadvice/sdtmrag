use anyhow::{Context, Result};
use calamine::{open_workbook, Data, Reader, Xlsx};
use std::path::Path;

pub struct SheetRecord {
    pub sheet: String,
    pub row: usize,
    /// (header, value) пары для непустых ячеек данных
    pub cells: Vec<(String, String)>,
}

fn cell_str(c: &Data) -> Option<String> {
    match c {
        Data::String(s) => {
            let t = s.trim();
            if t.is_empty() { None } else { Some(t.to_string()) }
        }
        Data::Float(f) => Some(trim_number(f.to_string())),
        Data::Int(i) => Some(i.to_string()),
        Data::Bool(b) => Some(b.to_string()),
        Data::DateTime(d) => Some(d.to_string()),
        Data::DateTimeIso(s) => Some(s.to_string()),
        Data::DurationIso(s) => Some(s.to_string()),
        Data::Error(e) => Some(format!("<error: {e}>")),
        Data::Empty => None,
    }
}

fn trim_number(s: String) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// Читает .xlsx/.xlsm: each sheet -> строки данных с заголовком.
/// Заголовок — первая строка листа с >= 2 непустыми ячейками.
pub fn read_xlsx(path: &Path) -> Result<Vec<SheetRecord>> {
    let mut wb: Xlsx<_> = open_workbook(path)
        .with_context(|| format!("cannot open xlsx: {}", path.display()))?;
    let mut out = Vec::new();
    for (sheet_name, range) in wb.worksheets() {
        let rows: Vec<Vec<Option<String>>> = range.rows().map(|r| r.iter().map(cell_str).collect()).collect();
        if rows.is_empty() {
            continue;
        }
        let header_idx = rows
            .iter()
            .take(8)
            .position(|r| r.iter().filter(|c| c.is_some()).count() >= 2);
        let Some(hi) = header_idx else {
            continue;
        };
        let header: Vec<String> = rows[hi]
            .iter()
            .enumerate()
            .map(|(i, c)| c.clone().unwrap_or_else(|| format!("col{i}")))
            .collect();
        for (idx, row) in rows[hi + 1..].iter().enumerate() {
            let cells: Vec<(String, String)> = header
                .iter()
                .enumerate()
                .filter_map(|(j, h)| {
                    row.get(j).and_then(|v| v.clone()).map(|v| (h.clone(), v))
                })
                .collect();
            if cells.iter().filter(|(_, v)| !v.is_empty()).count() >= 2 || !cells.is_empty() {
                out.push(SheetRecord {
                    sheet: sheet_name.clone(),
                    row: hi + 1 + idx + 1, // 1-based, чтобы совпадал с Excel
                    cells,
                });
            }
        }
    }
    Ok(out)
}