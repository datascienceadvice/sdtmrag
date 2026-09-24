//! Очистка текста, извлекаемого из PDF.
//!
//! `pdf-extract` отдаёт «сырой» поток, в котором:
//!
//! * постраничные колонтитулы и копирайты склеиваются с содержательными
//!   строками (например, `... CRF  nsortium, Inc. All rights reserved  Page 258 2021-11-29`);
//! * таблицы схлопываются в одну строку, где столбцы разделены сериями пробелов;
//! * оглавление попадает в поток как обычный текст и выглядит как заголовки
//!   разделов с точечными выносками;
//! * слова остаются разорванными переносом на границе строк;
//! * встречаются лигатуры, типографские тире/кавычки, неразрывные пробелы,
//!   soft hyphen и символы-заместители.
//!
//! Модуль приводит такой текст к виду, пригодному для чанкинга и эмбеддинга, и
//! возвращает [`CleanStats`] — статистику по каждому классу удалённого мусора.
//! Эти числа используются в отчётах об оценке качества RAG.

use crate::config::CleaningConfig;
use crate::extract::pdf::PdfPage;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Одна страница после очистки.
#[derive(Debug, Clone)]
pub struct CleanedPage {
    pub page: usize,
    pub text: String,
}

/// Статистика очистки одного документа (или всего корпуса).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct CleanStats {
    pub pages: usize,
    pub lines_in: usize,
    pub lines_out: usize,
    pub chars_in: usize,
    pub chars_out: usize,
    /// символы, удалённые inline-правилами (колонтитулы внутри строк)
    pub inline_scrubbed_chars: usize,
    /// замены типографских символов (лигатуры, тире, кавычки, nbsp)
    pub unicode_substitutions: usize,
    /// строки оглавления (точечные выноски, CONTENTS)
    pub toc_lines: usize,
    /// строки-колонтитулы (копирайт, номер страницы, дата, running header)
    pub boilerplate_lines: usize,
    /// повторяющиеся строки, оставленные только в первом вхождении
    pub repeated_lines: usize,
    /// склеек слов, разорванных переносом
    pub dehyphenations: usize,
    /// строк, распознанных как табличные и переразмеченных в `a | b | c`
    pub table_lines: usize,
}

impl CleanStats {
    pub fn merge(&mut self, other: &CleanStats) {
        self.pages += other.pages;
        self.lines_in += other.lines_in;
        self.lines_out += other.lines_out;
        self.chars_in += other.chars_in;
        self.chars_out += other.chars_out;
        self.inline_scrubbed_chars += other.inline_scrubbed_chars;
        self.unicode_substitutions += other.unicode_substitutions;
        self.toc_lines += other.toc_lines;
        self.boilerplate_lines += other.boilerplate_lines;
        self.repeated_lines += other.repeated_lines;
        self.dehyphenations += other.dehyphenations;
        self.table_lines += other.table_lines;
    }

    /// Доля удалённых символов.
    pub fn removed_ratio(&self) -> f64 {
        if self.chars_in == 0 {
            return 0.0;
        }
        1.0 - (self.chars_out as f64 / self.chars_in as f64)
    }

    pub fn removed_lines(&self) -> usize {
        self.lines_in.saturating_sub(self.lines_out)
    }
}

// ---------------------------------------------------------------------------
// Unicode
// ---------------------------------------------------------------------------

/// Типографский символ -> ASCII-эквивалент. `None` — символ оставляем как есть.
fn map_char(ch: char) -> Option<&'static str> {
    Some(match ch {
        '\u{00A0}' | '\u{1680}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => " ",
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}' | '\u{2212}' => "-",
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' | '\u{2039}' | '\u{203A}' => "'",
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{2033}' => "\"",
        '\u{2026}' => "...",
        // невидимые/служебные
        '\u{00AD}' | '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' | '\u{FFFD}' => "",
        // знаки охраны прав
        '\u{00A9}' | '\u{00AE}' | '\u{2122}' => "",
        // лигатуры
        '\u{FB00}' => "ff",
        '\u{FB01}' => "fi",
        '\u{FB02}' => "fl",
        '\u{FB03}' => "ffi",
        '\u{FB04}' => "ffl",
        // маркеры списков
        '\u{2022}' | '\u{00B7}' | '\u{25AA}' | '\u{25CF}' | '\u{25A0}' | '\u{25A1}' | '\u{25CB}' => "-",
        _ => return None,
    })
}

fn normalize_unicode(input: &str) -> (String, usize) {
    let mut out = String::with_capacity(input.len());
    let mut n = 0usize;
    for ch in input.chars() {
        match map_char(ch) {
            Some(rep) => {
                out.push_str(rep);
                n += 1;
            }
            None => out.push(ch),
        }
    }
    (out, n)
}

/// Приводит строку к «типографически нейтральному» виду: типографские тире и
/// кавычки -> ASCII, лигатуры раскрыты, невидимые символы удалены, пробелы
/// схлопнуты. Нужна там, где текст сравнивается между «сырым» и «очищенным»
/// индексом (например, при сопоставлении разделов в оценке качества).
pub fn normalize_typography(input: &str) -> String {
    let (mapped, _) = normalize_unicode(input);
    collapse_re().replace_all(mapped.trim(), " ").into_owned()
}

// ---------------------------------------------------------------------------
// Регулярные выражения
// ---------------------------------------------------------------------------

fn collapse_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\s+").expect("valid regex"))
}

/// Разрывы в 2+ пробела (или табуляция) — разделители столбцов в схлопнутой таблице.
fn gap_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\s{2,}|\t").expect("valid regex"))
}

/// Точечные выноски оглавления.
fn toc_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\.{3,}|(?:\s\.){3,}").expect("valid regex"))
}

fn toc_title_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)^\s*(?:table of\s+)?contents\s*$").expect("valid regex"))
}

/// Inline-мусор: колонтитулы, попавшие внутрь содержательных строк.
fn inline_patterns() -> &'static [Regex] {
    static RE: OnceLock<Vec<Regex>> = OnceLock::new();
    RE.get_or_init(|| {
        [
            // "CDISC Clinical Data Interchange Standards Consortium, Inc." и его обрывки ("nsortium, Inc.")
            r"(?i)[a-z]{0,4}onsortium,?\s*inc\.?\s*(?:all rights reserved)?",
            r"(?i)clinical data interchange standards consortium,?\s*inc\.?",
            r"(?i)all rights reserved\.?",
            // "Page 258 2021-11-29" / "Page 3"
            r"(?i)(?:^|\s)page\s+\d+(?:\s+\d{4}-\d{2}-\d{2})?",
            // бегущий колонтитул стандарта
            r"(?i)cdisc study data tabulation model implementation guide:[^(]*\(\s*\d+\.\d+\s*final[^)]*\)",
            r"(?i)cdisc study data tabulation model\s*\(\s*version\s*\d+\.\d+\s*final[^)]*\)",
        ]
        .iter()
        .map(|p| Regex::new(p).expect("valid regex"))
        .collect()
    })
}

// ---------------------------------------------------------------------------
// Детекторы мусора
// ---------------------------------------------------------------------------

/// Ключ строки для поиска повторов: регистр и пробелы не важны,
/// но цифры значимы (иначе разные строки данных склеятся).
fn repeat_key(line: &str) -> String {
    let mut s = String::with_capacity(line.len());
    let mut prev_space = false;
    for ch in line.trim().chars() {
        if ch.is_whitespace() {
            if !prev_space {
                s.push(' ');
                prev_space = true;
            }
        } else {
            prev_space = false;
            s.extend(ch.to_lowercase());
        }
    }
    s
}

/// Ключ с замаскированными цифрами — для шаблонных колонтитулов.
fn masked_key(line: &str) -> String {
    let mut s = String::with_capacity(line.len());
    let mut prev_space = false;
    for ch in line.trim().chars() {
        if ch.is_ascii_digit() {
            s.push('#');
            prev_space = false;
        } else if ch.is_whitespace() {
            if !prev_space {
                s.push(' ');
                prev_space = true;
            }
        } else {
            prev_space = false;
            s.extend(ch.to_lowercase());
        }
    }
    s
}

fn is_toc_line(line: &str) -> bool {
    if toc_title_re().is_match(line) {
        return true;
    }
    // выноски оглавления: не менее трёх точек подряд (или " . . . ")
    toc_re().is_match(line)
}

fn is_hard_boilerplate(masked: &str) -> bool {
    let m = masked.trim();
    if m.is_empty() {
        return true;
    }
    // чистый номер страницы
    if m.chars().all(|c| c == '#') {
        return true;
    }
    // ISO-дата выпуска документа (в этих PDF стоит в колонтитуле на каждой странице)
    if m.len() == 10 && m.chars().enumerate().all(|(i, c)| {
        if i == 4 || i == 7 {
            c == '-'
        } else {
            c == '#'
        }
    }) {
        return true;
    }
    if m.contains("all rights reserved")
        || m.contains("clinical data interchange standards consortium")
        || m.starts_with("cdisc study data tabulation model")
        || m.starts_with("page #")
    {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Основной проход
// ---------------------------------------------------------------------------

fn scrub_inline(line: &str) -> (String, usize) {
    let mut cur = line.to_string();
    let mut removed = 0usize;
    for re in inline_patterns() {
        if re.is_match(&cur) {
            let before = cur.chars().count();
            cur = re.replace_all(&cur, " ").into_owned();
            removed += before.saturating_sub(cur.chars().count());
        }
    }
    (cur, removed)
}

/// Переразметка табличной строки: `a  b  c` -> `a | b | c`.
/// Возвращает `(строка, была_ли_табличной)`.
fn render_line(line: &str, cfg: &CleaningConfig) -> (String, bool) {
    let trimmed = line.trim();
    if cfg.render_table_rows {
        let gaps = gap_re().find_iter(trimmed).count();
        if gaps >= cfg.table_gap_threshold {
            let cells: Vec<&str> = gap_re()
                .split(trimmed)
                .map(str::trim)
                .filter(|c| !c.is_empty())
                .collect();
            if cells.len() >= 2 {
                return (cells.join(" | "), true);
            }
        }
    }
    (collapse_re().replace_all(trimmed, " ").into_owned(), false)
}

/// Собирает словарь токенов, нужный для решения о переносе слов.
fn collect_vocab(lines: &[String], hyphenated: &mut HashSet<String>, plain: &mut HashSet<String>) {
    for line in lines {
        for tok in line.split(|c: char| !(c.is_alphanumeric() || c == '-' || c == '\'')) {
            if tok.len() < 4 {
                continue;
            }
            let t = tok.to_lowercase();
            if let Some(rest) = t.strip_prefix('\'') {
                if rest.is_empty() {
                    continue;
                }
            }
            if t.contains('-') {
                let core = t.trim_matches('-');
                if core.contains('-') && core.chars().all(|c| c.is_alphanumeric() || c == '-') {
                    hyphenated.insert(core.to_string());
                }
            } else if t.chars().all(|c| c.is_ascii_alphabetic()) {
                plain.insert(t);
            }
        }
    }
}

/// Нужно ли сохранить дефис при склейке строк `a` (заканчивается на `-`) и `b`.
///
/// Решение принимается по словарю корпуса: если дефисная форма встречается в
/// тексте как обычное слово (`non-standard`), а слитная — нет, значит дефис
/// смысловой. Иначе считаем, что это перенос, и склеиваем без дефиса.
fn keep_hyphen(a: &str, b: &str, hyphenated: &HashSet<String>, plain: &HashSet<String>) -> bool {
    let left: String = a
        .trim_end_matches('-')
        .rsplit(|c: char| !c.is_alphanumeric())
        .next()
        .unwrap_or("")
        .to_string();
    let right: String = b.chars().take_while(|c| c.is_alphanumeric()).collect();
    if left.len() < 2 || right.len() < 2 {
        return false;
    }
    let with = format!("{}-{}", left.to_lowercase(), right.to_lowercase());
    let without = format!("{}{}", left.to_lowercase(), right.to_lowercase());
    hyphenated.contains(&with) && !plain.contains(&without)
}

/// Признак того, что строку можно склеить со следующей (перенос слова).
fn ends_with_soft_hyphen(line: &str) -> bool {
    let t = line.trim_end();
    if !t.ends_with('-') {
        return false;
    }
    let stem = &t[..t.len() - 1];
    stem.chars().next_back().map(|c| c.is_alphabetic()).unwrap_or(false)
}

/// Основная функция: очищает страницы PDF-документа.
pub fn clean_pages(pages: &[PdfPage], cfg: &CleaningConfig) -> (Vec<CleanedPage>, CleanStats) {
    let mut stats = CleanStats {
        pages: pages.len(),
        ..Default::default()
    };

    if !cfg.enabled {
        let out: Vec<CleanedPage> = pages
            .iter()
            .map(|p| {
                stats.chars_in += p.text.chars().count();
                stats.lines_in += p.text.lines().count();
                stats.chars_out += p.text.chars().count();
                stats.lines_out += p.text.lines().count();
                CleanedPage {
                    page: p.page,
                    text: p.text.clone(),
                }
            })
            .collect();
        return (out, stats);
    }

    // ---- проход 1: нормализация, inline-чистка, отсев явного мусора ----
    let mut per_page: Vec<Vec<String>> = Vec::with_capacity(pages.len());
    // сколько страниц содержат строку с данным ключом
    let mut page_hits: HashMap<String, usize> = HashMap::new();

    for p in pages {
        stats.chars_in += p.text.chars().count();
        let mut kept: Vec<String> = Vec::new();
        for raw in p.text.lines() {
            stats.lines_in += 1;
            let (norm, unicode_subs) = if cfg.normalize_unicode {
                normalize_unicode(raw)
            } else {
                (raw.to_string(), 0)
            };
            stats.unicode_substitutions += unicode_subs;

            let (scrubbed, removed) = scrub_inline(&norm);
            stats.inline_scrubbed_chars += removed;

            let line = if cfg.render_table_rows {
                scrubbed.trim().to_string()
            } else {
                collapse_re().replace_all(scrubbed.trim(), " ").into_owned()
            };
            if line.is_empty() {
                continue;
            }
            if cfg.strip_toc && is_toc_line(&line) {
                stats.toc_lines += 1;
                continue;
            }
            if cfg.remove_boilerplate && is_hard_boilerplate(&masked_key(&line)) {
                stats.boilerplate_lines += 1;
                continue;
            }
            kept.push(line);
        }

        let mut seen: HashSet<String> = HashSet::new();
        for line in &kept {
            let k = repeat_key(line);
            if seen.insert(k.clone()) {
                *page_hits.entry(k).or_insert(0) += 1;
            }
        }
        per_page.push(kept);
    }

    // ---- порог повторяемости ----
    let (repeated_keys, threshold) = if cfg.remove_boilerplate {
        let ratio_threshold =
            (pages.len() as f64 * cfg.boilerplate_page_ratio).ceil() as usize;
        let threshold = ratio_threshold.max(cfg.boilerplate_min_pages);
        let keys: HashSet<String> = page_hits
            .iter()
            .filter(|(_, n)| **n >= threshold && **n >= 2)
            .map(|(k, _)| k.clone())
            .collect();
        (keys, threshold)
    } else {
        (HashSet::new(), usize::MAX)
    };
    let _ = threshold;

    // ---- проход 2: снятие повторов + словарь для переносов ----
    let mut emitted: HashSet<String> = HashSet::new();
    let mut filtered: Vec<Vec<String>> = Vec::with_capacity(per_page.len());
    for kept in per_page {
        let mut out = Vec::with_capacity(kept.len());
        for line in kept {
            let k = repeat_key(&line);
            if repeated_keys.contains(&k) {
                if emitted.insert(k) {
                    out.push(line); // первое вхождение оставляем
                } else {
                    stats.repeated_lines += 1;
                }
            } else {
                out.push(line);
            }
        }
        filtered.push(out);
    }

    let mut hyphenated: HashSet<String> = HashSet::new();
    let mut plain: HashSet<String> = HashSet::new();
    if cfg.dehyphenate {
        for lines in &filtered {
            collect_vocab(lines, &mut hyphenated, &mut plain);
        }
    }

    // ---- проход 3: склейка переносов и разметка таблиц ----
    let mut result: Vec<CleanedPage> = Vec::with_capacity(filtered.len());
    for (idx, lines) in filtered.into_iter().enumerate() {
        let mut merged: Vec<String> = Vec::with_capacity(lines.len());
        let mut pending: Option<String> = None;
        for line in lines {
            match pending.take() {
                None => {
                    if cfg.dehyphenate && ends_with_soft_hyphen(&line) {
                        pending = Some(line);
                    } else {
                        merged.push(line);
                    }
                }
                Some(prev) => {
                    if keep_hyphen(&prev, &line, &hyphenated, &plain) {
                        // дефис смысловой: строки склеиваем, дефис сохраняем
                        let joined = format!("{prev}{line}");
                        if ends_with_soft_hyphen(&joined) {
                            pending = Some(joined);
                        } else {
                            merged.push(joined);
                        }
                    } else {
                        // перенос слова: склеиваем без дефиса
                        stats.dehyphenations += 1;
                        let joined = format!("{}{}", prev.trim_end_matches('-'), line);
                        if ends_with_soft_hyphen(&joined) {
                            pending = Some(joined);
                        } else {
                            merged.push(joined);
                        }
                    }
                }
            }
        }
        if let Some(last) = pending {
            merged.push(last);
        }

        let mut out_lines: Vec<String> = Vec::with_capacity(merged.len());
        for line in merged {
            let (rendered, is_table) = render_line(&line, cfg);
            if rendered.is_empty() {
                continue;
            }
            if is_table {
                stats.table_lines += 1;
            }
            out_lines.push(rendered);
        }

        let text = out_lines.join("\n");
        stats.lines_out += out_lines.len();
        stats.chars_out += text.chars().count();
        result.push(CleanedPage {
            page: pages.get(idx).map(|p| p.page).unwrap_or(idx + 1),
            text,
        });
    }

    (result, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(n: usize, text: &str) -> PdfPage {
        PdfPage {
            page: n,
            text: text.to_string(),
        }
    }

    #[test]
    fn strips_inline_footer() {
        let pages = vec![page(
            1,
            "Row  ABC-01-101  BEREFID  SP02  BESPEC  Specimen Type  SPUTUM  CRF  nsortium, Inc. All rights reserved    Page 258 2021-11-29 The SUPPBE dataset is used.",
        )];
        let (out, stats) = clean_pages(&pages, &CleaningConfig::default());
        let text = &out[0].text;
        assert!(!text.contains("All rights reserved"), "{text}");
        assert!(!text.contains("Page 258"), "{text}");
        assert!(!text.contains("2021-11-29"), "{text}");
        assert!(text.contains("The SUPPBE dataset is used."), "{text}");
        assert!(stats.inline_scrubbed_chars > 0);
    }

    #[test]
    fn renders_table_rows_with_pipes() {
        let pages = vec![page(1, "Row  STUDYID  RDOMAIN  USUBJID\nplain prose line here")];
        let (out, stats) = clean_pages(&pages, &CleaningConfig::default());
        assert_eq!(out[0].text.lines().next().unwrap(), "Row | STUDYID | RDOMAIN | USUBJID");
        assert_eq!(out[0].text.lines().nth(1).unwrap(), "plain prose line here");
        assert_eq!(stats.table_lines, 1);
    }

    #[test]
    fn drops_toc_and_page_numbers() {
        let pages = vec![page(1, "1.1  PURPOSE ................. 5\n12\nreal content")];
        let (out, stats) = clean_pages(&pages, &CleaningConfig::default());
        assert_eq!(out[0].text, "real content");
        assert_eq!(stats.toc_lines, 1);
        assert_eq!(stats.boilerplate_lines, 1);
    }

    #[test]
    fn removes_repeated_lines_after_first_occurrence() {
        let mut pages = Vec::new();
        for i in 1..=6 {
            pages.push(page(i, &format!("running header text\ncontent number {i}")));
        }
        let (out, stats) = clean_pages(&pages, &CleaningConfig::default());
        let hits = out.iter().filter(|p| p.text.contains("running header text")).count();
        assert_eq!(hits, 1);
        assert_eq!(stats.repeated_lines, 5);
    }

    #[test]
    fn dehyphenates_soft_breaks_but_keeps_real_compounds() {
        let mut text = String::from("A non-standard baseline is defined here.\n");
        for _ in 0..6 {
            text.push_str("the non-\nstandard value needs the impl-\nementation guide\n");
        }
        let pages = vec![page(1, &text)];
        let (out, stats) = clean_pages(&pages, &CleaningConfig::default());
        assert!(out[0].text.contains("implementation"), "{}", out[0].text);
        assert!(out[0].text.contains("non-standard"), "{}", out[0].text);
        assert!(!out[0].text.contains("non-\nstandard"), "{}", out[0].text);
        assert_eq!(stats.dehyphenations, 6);
    }

    #[test]
    fn normalizes_typography() {
        let pages = vec![page(1, "The \u{201C}value\u{201D} \u{2013} 5\u{2013}10 \u{FB01}le \u{00A9}2021")];
        let (out, stats) = clean_pages(&pages, &CleaningConfig::default());
        assert_eq!(out[0].text, "The \"value\" - 5-10 file 2021");
        assert!(stats.unicode_substitutions >= 5);
    }
}
