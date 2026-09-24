use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use sdtmrag::config::Config;
use sdtmrag::embed::Embedder;
use serde::Serialize;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "sdtmrag", version, about = "RAG по документации SDTM/CDISC (локальные эмбеддинги + DeepSeek)")]
struct Cli {
    /// путь к config.toml
    #[arg(long, global = true)]
    config: Option<String>,
    /// переопределить путь к индексу (data/index.sqlite)
    #[arg(long, global = true)]
    store: Option<String>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Построить индекс по документам из inputs (dicts по умолчанию)
    Index {
        /// директория с исходниками (.pdf/.xlsx/.xlsm)
        #[arg(long)]
        inputs: Option<String>,
        /// пересобрать индекс с нуля
        #[arg(long)]
        rebuild: bool,
    },
    /// Найти ответ по вопросу (retrieval + реранк + генерация DeepSeek)
    Query {
        #[arg(required = true)]
        question: String,
        /// сколько источников показать
        #[arg(long, default_value_t = 6)]
        top_k: usize,
        /// отключить реранк
        #[arg(long)]
        no_rerank: bool,
        /// только поиск, без генерации DeepSeek
        #[arg(long)]
        no_generate: bool,
        /// вывести результат как JSON
        #[arg(long)]
        json: bool,
    },
    /// Показать, что даёт очистка PDF-текста (без построения индекса)
    Clean {
        /// директория с исходниками
        #[arg(long)]
        inputs: Option<String>,
        /// сохранить очищенный текст в эту директорию (по файлу на документ)
        #[arg(long)]
        out: Option<String>,
        /// вывести статистику как JSON
        #[arg(long)]
        json: bool,
        /// обработать только первые N файлов
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Оценить качество RAG по эталонному набору вопросов
    Eval {
        /// файлы «золотого» набора (JSON или JSONL); можно указать несколько раз
        #[arg(long = "gold", required = true)]
        gold: Vec<String>,
        /// индекс(ы) для сравнения в формате <путь>[=<метка>]; можно несколько раз
        #[arg(long = "store")]
        stores: Vec<String>,
        /// значения k для метрик
        #[arg(long = "k", value_delimiter = ',', default_value = "1,3,5,10")]
        ks: Vec<usize>,
        /// отключить реранк
        #[arg(long)]
        no_rerank: bool,
        /// сгенерировать ответы и оценить их LLM-судьёй (нужен DEEPSEEK_API_KEY)
        #[arg(long)]
        judge: bool,
        /// сколько выдержек подавать в генерацию
        #[arg(long, default_value_t = 6)]
        context_k: usize,
        /// каталог для report.json / report.md
        #[arg(long, default_value = "eval/reports")]
        out: Option<String>,
        /// ограничиться первыми N вопросами
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Статистика по индексу
    Stats {},
    /// Список источников в индексе
    ListDocs {},
    /// Запустить REST API (axum)
    Serve {
        #[arg(long)]
        bind: Option<String>,
        #[arg(long)]
        port: Option<u16>,
    },
}

#[derive(Serialize)]
struct QueryJson {
    question: String,
    answer: Option<String>,
    model: String,
    sources: Vec<SourceJson>,
}

#[derive(Serialize)]
struct SourceJson {
    score: f32,
    source: String,
    section: Option<String>,
    page: Option<i64>,
    sheet: Option<String>,
    row: Option<i64>,
    text: String,
}

fn init_logging() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).try_init();
}

fn embedder_for(cfg: &Config) -> Result<Embedder> {
    Embedder::from_config(cfg).context("failed to init embedding model")
}

fn rerank_enabled(cfg: &Config, no_rerank: bool) -> bool {
    cfg.rerank.enabled && !no_rerank
}

/// `<путь>[=<метка>]` -> (путь, метка)
fn parse_store_spec(spec: &str) -> (PathBuf, String) {
    match spec.split_once('=') {
        Some((p, label)) => (PathBuf::from(p), label.to_string()),
        None => {
            let path = PathBuf::from(spec);
            let label = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| spec.to_string());
            (path, label)
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_logging();
    let cli = Cli::parse();

    let mut cfg = Config::load(cli.config.as_deref())?;
    if let Some(s) = &cli.store {
        cfg.store_path = PathBuf::from(s);
    }

    match cli.cmd {
        Cmd::Index { inputs, rebuild } => {
            if let Some(i) = inputs {
                cfg.inputs = PathBuf::from(i);
            }
            let stats = sdtmrag::indexing::build_index(&cfg, rebuild)?;
            println!(
                "index built: {} file(s), {} chunks stored in {} ({:.1}s, embedding {:.1}s)",
                stats.files,
                stats.chunks,
                cfg.store_path.display(),
                stats.total_secs,
                stats.embed_secs
            );
            if cfg.cleaning.enabled {
                println!(
                    "text cleaning: chars {} -> {} (-{:.1}%), boilerplate {} lines, repeats {}, toc {}, dehyphenated {}, table rows {}",
                    stats.cleaning.chars_in,
                    stats.cleaning.chars_out,
                    stats.cleaning.removed_ratio() * 100.0,
                    stats.cleaning.boilerplate_lines,
                    stats.cleaning.repeated_lines,
                    stats.cleaning.toc_lines,
                    stats.cleaning.dehyphenations,
                    stats.cleaning.table_lines
                );
            } else {
                println!("text cleaning: disabled");
            }
        }
        Cmd::Query { question, top_k, no_rerank, no_generate, json } => {
            let store = sdtmrag::store::Store::open(&cfg.store_path, false)?;
            if store.count()? == 0 {
                anyhow::bail!("index is empty — run `sdtmrag index` first");
            }
            // модель эмбеддингов берём из индекса: он знает, какой моделью построен
            store.apply_embedding_meta(&mut cfg)?;
            let mut embedder = embedder_for(&cfg)?;

            let mut reranker = if rerank_enabled(&cfg, no_rerank) {
                Some(sdtmrag::embed::Reranker::new(
                    &cfg.rerank.model,
                    &cfg.models_dir,
                    true,
                    cfg.rerank.intra_threads,
                    cfg.rerank.max_length,
                )?)
            } else {
                None
            };
            let retrieved = sdtmrag::pipeline::search(
                &store,
                &mut embedder,
                &question,
                &cfg.rerank,
                top_k.max(cfg.rerank.final_top_k),
                reranker.as_mut(),
            )?;
            let shown = retrieved.into_iter().take(top_k).collect::<Vec<_>>();

            let (answer, model): (Option<String>, String) = if no_generate {
                (None, String::from("-"))
            } else {
                cfg.require_api_key()?;
                let client = sdtmrag::generation::DeepSeekClient::new(&cfg.deepseek)?;
                let a = sdtmrag::pipeline::generate(&client, &question, &shown).await?;
                (Some(a), client.model().to_string())
            };

            if json {
                let out = QueryJson {
                    question,
                    model,
                    answer,
                    sources: shown
                        .iter()
                        .map(|r| SourceJson {
                            score: r.score,
                            source: r.chunk.source.clone(),
                            section: r.chunk.section.clone(),
                            page: r.chunk.page,
                            sheet: r.chunk.sheet.clone(),
                            row: r.chunk.row,
                            text: r.chunk.text.clone(),
                        })
                        .collect(),
                };
                println!("{}", serde_json::to_string_pretty(&out)?);
            } else {
                if let Some(a) = &answer {
                    println!("{a}\n");
                }
                println!("--- Sources (извлечённые выдержки) ---");
                for (i, r) in shown.iter().enumerate() {
                    let mut loc: Vec<String> = vec![r.chunk.source.clone()];
                    if let Some(p) = r.chunk.page {
                        loc.push(format!("page {p}"));
                    }
                    if let Some(s) = &r.chunk.section {
                        loc.push(s.clone());
                    }
                    if let Some(sh) = &r.chunk.sheet {
                        loc.push(sh.clone());
                    }
                    if let Some(row) = r.chunk.row {
                        loc.push(format!("row {row}"));
                    }
                    println!("[{}] score={:.3} {loc:?}", i + 1, r.score);
                    let preview: String = r.chunk.text.chars().take(400).collect();
                    println!("    {}", preview.replace('\n', " "));
                    println!();
                }
            }
        }
        Cmd::Clean { inputs, out, json, limit } => {
            if let Some(i) = inputs {
                cfg.inputs = PathBuf::from(i);
            }
            let files = sdtmrag::extract::supported_files(&cfg.inputs)?;
            let files: Vec<PathBuf> = match limit {
                Some(n) => files.into_iter().take(n).collect(),
                None => files,
            };
            let mut total = sdtmrag::clean::CleanStats::default();
            let mut per_file: Vec<(String, sdtmrag::clean::CleanStats)> = Vec::new();
            if let Some(dir) = &out {
                std::fs::create_dir_all(dir)
                    .with_context(|| format!("cannot create output dir: {dir}"))?;
            }
            for path in &files {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string());
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
                if ext != "pdf" {
                    continue;
                }
                let pages = sdtmrag::extract::pdf::extract_pdf_pages(path)?;
                let (cleaned, stats) = sdtmrag::clean::clean_pages(&pages, &cfg.cleaning);
                if let Some(dir) = &out {
                    let text: String = cleaned
                        .iter()
                        .map(|p| format!("\n===== page {} =====\n{}", p.page, p.text))
                        .collect();
                    let dest = PathBuf::from(dir).join(format!("{name}.txt"));
                    std::fs::write(&dest, text)?;
                    eprintln!("[clean] wrote {}", dest.display());
                }
                if !json {
                    println!(
                        "{name}: pages={} chars {} -> {} (-{:.1}%) | boilerplate={} repeats={} toc={} dehyphen={} tables={} unicode={} inline_chars={}",
                        stats.pages,
                        stats.chars_in,
                        stats.chars_out,
                        stats.removed_ratio() * 100.0,
                        stats.boilerplate_lines,
                        stats.repeated_lines,
                        stats.toc_lines,
                        stats.dehyphenations,
                        stats.table_lines,
                        stats.unicode_substitutions,
                        stats.inline_scrubbed_chars
                    );
                }
                total.merge(&stats);
                per_file.push((name, stats));
            }
            if json {
                #[derive(Serialize)]
                struct Out {
                    total: sdtmrag::clean::CleanStats,
                    files: Vec<serde_json::Value>,
                }
                let out = Out {
                    total: total.clone(),
                    files: per_file
                        .iter()
                        .map(|(n, s)| {
                            serde_json::json!({
                                "file": n,
                                "stats": s,
                            })
                        })
                        .collect(),
                };
                println!("{}", serde_json::to_string_pretty(&out)?);
            } else {
                println!(
                    "TOTAL: chars {} -> {} (-{:.1}%), removed {} lines of {} ({:.1}%)",
                    total.chars_in,
                    total.chars_out,
                    total.removed_ratio() * 100.0,
                    total.removed_lines(),
                    total.lines_in,
                    total.removed_lines() as f64 / total.lines_in.max(1) as f64 * 100.0
                );
            }
        }
        Cmd::Eval { gold, stores, ks, no_rerank, judge, context_k, out, limit } => {
            let gold_paths: Vec<PathBuf> = gold.iter().map(PathBuf::from).collect();
            let items = sdtmrag::eval::load_gold(&gold_paths)?;
            if items.is_empty() {
                anyhow::bail!("gold set is empty");
            }
            eprintln!(
                "[eval] {} questions ({} ru), ks={:?}, rerank={}, judge={}",
                items.len(),
                items.iter().filter(|i| i.lang == "ru").count(),
                ks,
                !no_rerank && cfg.rerank.enabled,
                judge
            );

            let specs: Vec<String> = if stores.is_empty() {
                vec![cfg.store_path.display().to_string()]
            } else {
                stores.clone()
            };

            let client = if judge {
                cfg.require_api_key()?;
                Some(sdtmrag::generation::DeepSeekClient::new(&cfg.deepseek)?)
            } else {
                None
            };

            let opts = sdtmrag::eval::RunOptions {
                ks: ks.clone(),
                rerank: !no_rerank,
                judge,
                context_k,
                limit,
            };

            let mut configs = Vec::new();
            for spec in &specs {
                let (path, label) = parse_store_spec(spec);
                eprintln!("[eval] === {label} ({}) ===", path.display());
                let res = sdtmrag::eval::evaluate_store(
                    &cfg,
                    &path,
                    &label,
                    &items,
                    &opts,
                    client.as_ref(),
                )
                .await?;
                configs.push(res);
            }

            let report = sdtmrag::eval::EvalReport {
                generated_at: sdtmrag::utc_now_string(),
                gold_items: items.len(),
                gold_files: gold_paths
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect(),
                ks,
                store_path: specs.join(", "),
                configs,
            };

            let out_dir = PathBuf::from(out.unwrap_or_else(|| "eval/reports".into()));
            let stamp = report
                .generated_at
                .replace([':', ' '], "-");
            let out_dir = out_dir.join(stamp);
            let (json_path, md_path) = sdtmrag::eval::write_report(&report, &out_dir)?;
            println!("{}", sdtmrag::eval::render_markdown(&report));
            eprintln!("[eval] wrote {}", json_path.display());
            eprintln!("[eval] wrote {}", md_path.display());
        }
        Cmd::Stats {} => {
            let store = sdtmrag::store::Store::open(&cfg.store_path, false)?;
            println!("store:            {}", cfg.store_path.display());
            println!("chunks:           {}", store.count()?);
            for (k, v) in store.all_meta()? {
                let v = if v.chars().count() > 160 {
                    format!("{}…", v.chars().take(160).collect::<String>())
                } else {
                    v
                };
                println!("{k:18}{v}");
            }
            println!("--- sources ---");
            for (src, n) in store.sources()? {
                println!("  {}\t{n}", src);
            }
        }
        Cmd::ListDocs {} => {
            let store = sdtmrag::store::Store::open(&cfg.store_path, false)?;
            for (src, n) in store.sources()? {
                println!("{src}\t{n}");
            }
        }
        Cmd::Serve { bind, port } => {
            if let Some(b) = bind {
                cfg.server.bind = b;
            }
            if let Some(p) = port {
                cfg.server.port = p;
            }
            let store = sdtmrag::store::Store::open(&cfg.store_path, false)?;
            if store.count()? == 0 {
                anyhow::bail!("index is empty — run `sdtmrag index` first");
            }
            #[cfg(feature = "server")]
            {
                let client = sdtmrag::generation::DeepSeekClient::new(&cfg.deepseek)?;
                sdtmrag::serve::serve(cfg, store, client).await?;
            }
            #[cfg(not(feature = "server"))]
            {
                let _ = store;
                anyhow::bail!("build with `--features server` to use `serve`");
            }
        }
    }
    Ok(())
}
