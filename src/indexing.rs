use crate::chunk::{chunk_pdf, chunk_xlsx, ChunkRecord};
use crate::clean::{clean_pages, CleanStats};
use crate::config::Config;
use crate::embed::Embedder;
use crate::extract;
use crate::extract::pdf::extract_pdf_pages;
use crate::extract::xlsx::read_xlsx;
use crate::store::Store;
use anyhow::{bail, Context, Result};
use std::time::Instant;

pub struct IndexStats {
    pub files: usize,
    pub chunks: usize,
    pub cleaning: CleanStats,
    pub embed_secs: f64,
    pub total_secs: f64,
}

pub fn build_index(cfg: &Config, rebuild: bool) -> Result<IndexStats> {
    let t_start = Instant::now();
    let files = extract::supported_files(&cfg.inputs)?;
    if files.is_empty() {
        bail!("no supported files (.pdf/.xlsx/.xlsm) found in {}", cfg.inputs.display());
    }
    warn_on_encoder_window(cfg);

    eprintln!("[index] extracting {} files from {}", files.len(), cfg.inputs.display());
    if cfg.cleaning.enabled {
        eprintln!(
            "[index] pdf cleaning: on (unicode={}, dehyphenate={}, toc={}, boilerplate={}, tables={})",
            cfg.cleaning.normalize_unicode,
            cfg.cleaning.dehyphenate,
            cfg.cleaning.strip_toc,
            cfg.cleaning.remove_boilerplate,
            cfg.cleaning.render_table_rows
        );
    } else {
        eprintln!("[index] pdf cleaning: OFF (raw extracted text)");
    }

    let mut chunks: Vec<ChunkRecord> = Vec::new();
    let mut cleaning = CleanStats::default();
    for path in &files {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        match ext.as_str() {
            "pdf" => {
                eprintln!("[index]   pdf: {name} ...");
                let raw_pages = extract_pdf_pages(path)?;
                let (pages, stats) = clean_pages(&raw_pages, &cfg.cleaning);
                if cfg.cleaning.enabled {
                    eprintln!(
                        "[index]         clean: {} pages, chars {} -> {} (-{:.1}%), \
                         boilerplate {}, repeats {}, toc {}, dehyphen {}, tables {}",
                        stats.pages,
                        stats.chars_in,
                        stats.chars_out,
                        stats.removed_ratio() * 100.0,
                        stats.boilerplate_lines,
                        stats.repeated_lines,
                        stats.toc_lines,
                        stats.dehyphenations,
                        stats.table_lines
                    );
                }
                cleaning.merge(&stats);
                let pdf_chunks = chunk_pdf(&pages, &name, &cfg.chunking)?;
                eprintln!("[index]         {} pages -> {} chunks", pages.len(), pdf_chunks.len());
                chunks.extend(pdf_chunks);
            }
            "xlsx" | "xlsm" => {
                eprintln!("[index]   xlsx: {name} ...");
                let rows = read_xlsx(path)?;
                let x_chunks = chunk_xlsx(&rows, &name);
                eprintln!("[index]         {} rows -> {} chunks", rows.len(), x_chunks.len());
                chunks.extend(x_chunks);
            }
            other => {
                bail!("unsupported file extension: {other} ({})", path.display());
            }
        }
    }

    if chunks.is_empty() {
        bail!("no chunks produced — extracted text is empty?");
    }

    let batch_size = cfg.embedding.batch_size.max(1);
    eprintln!(
        "[index] embedding {} chunks with {} (threads={}, batch={})",
        chunks.len(),
        cfg.embedding.model,
        cfg.embedding.intra_threads,
        batch_size
    );
    let mut embedder = Embedder::from_config(cfg)
        .with_context(|| format!("failed to init embedding model {}", cfg.embedding.model))?;
    eprintln!("[index] model loaded, starting embedding...");

    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let total = texts.len();
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(total);
    let t0 = Instant::now();
    let mut done = 0usize;
    for (batch_idx, batch) in texts.chunks(batch_size).enumerate() {
        let emb = embedder.embed_documents(batch)?;
        embeddings.extend(emb);
        done += batch.len();
        let elapsed = t0.elapsed().as_secs_f64();
        let rate = done as f64 / elapsed;
        let remaining = (total - done) as f64 / rate.max(1.0);
        eprintln!(
            "[index]   {done}/{total} chunks ({:.0}%) | {:.1} ch/s | ~{:.0}s left",
            done as f64 / total as f64 * 100.0,
            rate,
            remaining
        );
        let _ = batch_idx;
    }
    let elapsed = t0.elapsed();
    eprintln!("[index] embedding done in {:.1}s", elapsed.as_secs_f64());

    eprintln!("[index] writing to SQLite ...");
    let store = Store::open(&cfg.store_path, rebuild)?;
    let batch: Vec<(ChunkRecord, Vec<f32>)> = chunks.into_iter().zip(embeddings).collect();
    store.insert_batch(&batch)?;
    write_meta(&store, cfg, &cleaning, &files)?;
    let total_chunks = store.count()?;
    eprintln!("[index] stored {total_chunks} chunks in {}", cfg.store_path.display());

    Ok(IndexStats {
        files: files.len(),
        chunks: total_chunks as usize,
        cleaning,
        embed_secs: elapsed.as_secs_f64(),
        total_secs: t_start.elapsed().as_secs_f64(),
    })
}

/// Фиксирует в индексе всё, что нужно для воспроизводимости прогонов:
/// модель эмбеддинга, параметры чанкинга/очистки и статистику очистки.
fn write_meta(
    store: &Store,
    cfg: &Config,
    cleaning: &CleanStats,
    files: &[std::path::PathBuf],
) -> Result<()> {
    let set = |k: &str, v: String| -> Result<()> { store.set_meta(k, &v) };
    set("sdtmrag_version", env!("CARGO_PKG_VERSION").to_string())?;
    set("built_at", crate::utc_now_string())?;
    set("inputs", cfg.inputs.display().to_string())?;
    set("embedding_model", cfg.embedding.model.clone())?;
    set(
        "embedding_max_length",
        cfg.embedding.max_length.unwrap_or(512).to_string(),
    )?;
    set("chunking", serde_json::to_string(&cfg.chunking)?)?;
    set("cleaning", serde_json::to_string(&cfg.cleaning)?)?;
    set("cleaning_stats", serde_json::to_string(cleaning)?)?;
    let names: Vec<String> = files
        .iter()
        .map(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string())
        })
        .collect();
    set("files", serde_json::to_string(&names)?)?;
    Ok(())
}

/// Чанк длиннее окна энкодера молча обрезается при эмбеддинге —
/// предупреждаем, потому что это типичная причина «пропавшего» контекста.
pub fn warn_on_encoder_window(cfg: &Config) {
    let max_len = cfg.embedding.max_length.unwrap_or(512);
    // грубая оценка: 1 токен ≈ 4 символа
    let est_chars = max_len * 4;
    let target = (cfg.chunking.target_tokens * 4).min(cfg.chunking.max_chunk_chars);
    if target > est_chars {
        eprintln!(
            "[index] WARNING: target chunk size ~{target} chars exceeds the embedding window \
             (~{est_chars} chars = {max_len} tokens) — long chunks will be truncated at embedding time. \
             Consider chunking.target_tokens <= {}.",
            max_len
        );
    }
}