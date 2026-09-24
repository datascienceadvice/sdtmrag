//! Оценка качества RAG: эталонный набор вопросов, retrieval-метрики и
//! (опционально) оценка ответов LLM-судьёй.
//!
//! Метрики считаются по *локаторам* — устойчивым во времени описаниям места в
//! документе (файл + страница / лист + строка / раздел). Они не зависят от
//! того, как именно чанкер нарезал текст, поэтому один и тот же «золотой»
//! набор пригоден для сравнения разных конфигураций индекса.

use crate::config::Config;
use crate::embed::{Embedder, Reranker};
use crate::generation::DeepSeekClient;
use crate::pipeline::{self, Retrieved};
use crate::store::Store;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const DEFAULT_KS: &[usize] = &[1, 3, 5, 10];

// ---------------------------------------------------------------------------
// Эталонный набор
// ---------------------------------------------------------------------------

/// Локатор фрагмента, который считается релевантным ответом на вопрос.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Locator {
    /// имя файла-источника (как в индексе)
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sheet: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row: Option<i64>,
    /// дополнительная проверка: подстрока обязана встретиться в тексте чанка
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contains: Option<String>,
}

fn default_lang() -> String {
    "en".to_string()
}

fn default_true() -> bool {
    true
}

/// Один эталонный вопрос.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GoldItem {
    pub id: String,
    pub question: String,
    #[serde(default = "default_lang")]
    pub lang: String,
    #[serde(default)]
    pub topic: Option<String>,
    /// `false` — вопрос вне области документации, система обязана отказаться отвечать
    #[serde(default = "default_true")]
    pub answerable: bool,
    /// попадание в любой из локаторов считается успехом
    #[serde(default)]
    pub expected: Vec<Locator>,
    #[serde(default)]
    pub reference_answer: Option<String>,
    /// вопрос сгенерирован автоматически и ещё не проверен экспертом
    #[serde(default)]
    pub needs_review: bool,
    #[serde(default)]
    pub notes: Option<String>,
}

/// Читает «золотой» набор: JSON-массив или JSONL.
pub fn load_gold(paths: &[PathBuf]) -> Result<Vec<GoldItem>> {
    let mut out = Vec::new();
    for path in paths {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read gold set: {}", path.display()))?;
        let trimmed = text.trim_start();
        if trimmed.starts_with('[') {
            let items: Vec<GoldItem> = serde_json::from_str(trimmed)
                .with_context(|| format!("invalid JSON array in {}", path.display()))?;
            out.extend(items);
        } else {
            for (i, line) in text.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() || line.starts_with("//") {
                    continue;
                }
                let item: GoldItem = serde_json::from_str(line).with_context(|| {
                    format!("invalid JSONL at {}:{}", path.display(), i + 1)
                })?;
                out.extend(std::iter::once(item));
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    for item in &out {
        if !seen.insert(item.id.clone()) {
            anyhow::bail!("duplicate gold item id: {}", item.id);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Сопоставление
// ---------------------------------------------------------------------------

/// Нормализация для сопоставления локаторов: типографика -> ASCII, регистр и
/// пробелы не важны. Благодаря этому один и тот же «золотой» набор применим и к
/// «сырому», и к очищенному индексу (в последнем типографские тире уже заменены).
fn norm(s: &str) -> String {
    crate::clean::normalize_typography(s).to_lowercase()
}

impl Locator {
    pub fn matches(&self, c: &crate::store::StoredChunk) -> bool {
        if !c.source.eq_ignore_ascii_case(&self.source) {
            return false;
        }
        if let Some(p) = self.page {
            if c.page != Some(p) {
                return false;
            }
        }
        if let Some(s) = &self.sheet {
            match &c.sheet {
                Some(cs) if cs.eq_ignore_ascii_case(s) => {}
                _ => return false,
            }
        }
        if let Some(r) = self.row {
            if c.row != Some(r) {
                return false;
            }
        }
        if let Some(sec) = &self.section {
            match &c.section {
                Some(cs) if norm(cs).contains(&norm(sec)) => {}
                _ => return false,
            }
        }
        if let Some(sub) = &self.contains {
            if !norm(&c.text).contains(&norm(sub)) {
                return false;
            }
        }
        true
    }
}

impl GoldItem {
    fn matches(&self, c: &crate::store::StoredChunk) -> bool {
        self.expected.iter().any(|l| l.matches(c))
    }
}

// ---------------------------------------------------------------------------
// Метрики
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize)]
pub struct Aggregate {
    pub n: usize,
    pub mrr: f64,
    pub hit_at_k: BTreeMap<usize, f64>,
    pub recall_at_k: BTreeMap<usize, f64>,
    pub precision_at_k: BTreeMap<usize, f64>,
    pub ndcg_at_k: BTreeMap<usize, f64>,
}

#[derive(Debug, Clone, Default)] 
struct Accum {
    n: usize,
    rr: f64,
    hit: BTreeMap<usize, f64>,
    recall: BTreeMap<usize, f64>,
    precision: BTreeMap<usize, f64>,
    ndcg: BTreeMap<usize, f64>,
}

impl Accum {
    fn new(ks: &[usize]) -> Self {
        let mut a = Accum::default();
        for &k in ks {
            a.hit.insert(k, 0.0);
            a.recall.insert(k, 0.0);
            a.precision.insert(k, 0.0);
            a.ndcg.insert(k, 0.0);
        }
        a
    }

    fn push(&mut self, r: &RetrievalScore) {
        self.n += 1;
        self.rr += r.reciprocal_rank;
        for (k, v) in &r.hit_at_k {
            *self.hit.entry(*k).or_insert(0.0) += *v;
        }
        for (k, v) in &r.recall_at_k {
            *self.recall.entry(*k).or_insert(0.0) += *v;
        }
        for (k, v) in &r.precision_at_k {
            *self.precision.entry(*k).or_insert(0.0) += *v;
        }
        for (k, v) in &r.ndcg_at_k {
            *self.ndcg.entry(*k).or_insert(0.0) += *v;
        }
    }

    fn finish(&self) -> Aggregate {
        let d = |x: f64| if self.n == 0 { 0.0 } else { x / self.n as f64 };
        Aggregate {
            n: self.n,
            mrr: d(self.rr),
            hit_at_k: self.hit.iter().map(|(k, v)| (*k, d(*v))).collect(),
            recall_at_k: self.recall.iter().map(|(k, v)| (*k, d(*v))).collect(),
            precision_at_k: self.precision.iter().map(|(k, v)| (*k, d(*v))).collect(),
            ndcg_at_k: self.ndcg.iter().map(|(k, v)| (*k, d(*v))).collect(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RetrievalScore {
    pub reciprocal_rank: f64,
    pub first_relevant_rank: Option<usize>,
    pub hits_total: usize,
    pub hit_at_k: BTreeMap<usize, f64>,
    pub recall_at_k: BTreeMap<usize, f64>,
    pub precision_at_k: BTreeMap<usize, f64>,
    pub ndcg_at_k: BTreeMap<usize, f64>,
}

/// Считает метрики одного вопроса по ранжированному списку чанков.
pub fn score_retrieval(item: &GoldItem, retrieved: &[Retrieved], ks: &[usize]) -> RetrievalScore {
    let n_loc = item.expected.len().max(1) as f64;
    let mut out = RetrievalScore::default();

    let first = retrieved
        .iter()
        .position(|r| item.matches(&r.chunk))
        .map(|i| i + 1);
    out.first_relevant_rank = first;
    out.reciprocal_rank = first.map(|r| 1.0 / r as f64).unwrap_or(0.0);

    for &k in ks {
        let top = &retrieved[..retrieved.len().min(k)];
        // сколько уникальных локаторов закрыто
        let closed = item
            .expected
            .iter()
            .filter(|l| top.iter().any(|r| l.matches(&r.chunk)))
            .count();
        // сколько чанков в топ-k релевантны
        let rel = top.iter().filter(|r| item.matches(&r.chunk)).count();

        out.hit_at_k.insert(k, if closed > 0 { 1.0 } else { 0.0 });
        out.recall_at_k.insert(k, closed as f64 / n_loc);
        out.precision_at_k.insert(k, if k == 0 { 0.0 } else { rel as f64 / k as f64 });

        // nDCG с бинарной релевантностью.
        //
        // Важная деталь: эталон задан локаторами, и один локатор могут закрывать
        // несколько разных чанков. Если считать выигрыш по чанкам, DCG может
        // превысить идеальный и nDCG окажется больше 1. Поэтому выигрыш
        // начисляется только за *новый* локатор, впервые закрытый на этой позиции;
        // верхняя граница — все локаторы закрыты в первой позиции.
        let mut seen: Vec<usize> = Vec::new();
        let mut dcg = 0.0f64;
        for (i, r) in top.iter().enumerate() {
            let mut new_locators = 0usize;
            for (li, l) in item.expected.iter().enumerate() {
                if !seen.contains(&li) && l.matches(&r.chunk) {
                    seen.push(li);
                    new_locators += 1;
                }
            }
            if new_locators > 0 {
                dcg += new_locators as f64 / ((i as f64) + 2.0).log2();
            }
        }
        let idcg = n_loc;
        out.ndcg_at_k
            .insert(k, if idcg > 0.0 { (dcg / idcg).min(1.0) } else { 0.0 });
    }
    out.hits_total = retrieved.iter().filter(|r| item.matches(&r.chunk)).count();
    out
}

// ---------------------------------------------------------------------------
// Оценка ответов LLM-судьёй
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JudgeScore {
    /// доля утверждений ответа, подтверждённых выдержками
    pub faithfulness: f64,
    /// корректность ссылок [n] на выдержки
    pub citation_accuracy: f64,
    /// соответствие эталонному ответу
    pub correctness: f64,
    /// для вопросов вне документации — правильно ли система отказалась отвечать
    #[serde(default)]
    pub refusal_correct: Option<bool>,
    #[serde(default)]
    pub rationale: String,
}

const JUDGE_SYSTEM: &str = "You are a strict evaluator of retrieval-augmented answers about \
the CDISC SDTM standard and the SDTM Implementation Guide. \
You judge whether an answer is supported by the supplied documentation excerpts. \
Return JSON only, with no markdown fences.";

fn judge_prompt(item: &GoldItem, retrieved: &[Retrieved], answer: &str) -> String {
    let excerpts = pipeline::render_context(retrieved);
    let reference = item
        .reference_answer
        .as_deref()
        .unwrap_or("(not provided)");
    format!(
        "QUESTION:\n{question}\n\nREFERENCE ANSWER (ground truth, may be incomplete):\n{reference}\n\n\
         DOCUMENTATION EXCERPTS GIVEN TO THE SYSTEM:\n{excerpts}\n\nSYSTEM ANSWER:\n{answer}\n\n\
         The question is {answerable}. Score each criterion in [0,1]:\n\
         - faithfulness: fraction of factual claims in the answer that are directly supported by the excerpts;\n\
         - citation_accuracy: whether the excerpt markers cited ([1], [2], ...) actually support the claims they are attached to;\n\
         - correctness: agreement with the reference answer (if the question is NOT answerable from the documentation, \
         correctness = 1 when the system clearly states the information was not found, and 0 when it invents an answer).\n\
         Also set refusal_correct: true/false when the question is not answerable from the documentation, otherwise null.\n\n\
         Respond exactly as: {{\"faithfulness\": 0.0, \"citation_accuracy\": 0.0, \"correctness\": 0.0, \
         \"refusal_correct\": null, \"rationale\": \"...\"}}",
        question = item.question,
        reference = reference,
        excerpts = excerpts,
        answer = answer,
        answerable = if item.answerable {
            "answerable from the documentation"
        } else {
            "NOT answerable from the documentation"
        },
    )
}

fn parse_judge_json(raw: &str) -> Result<JudgeScore> {
    let start = raw.find('{').context("judge returned no JSON object")?;
    let end = raw.rfind('}').context("judge returned no JSON object")?;
    let slice = &raw[start..=end];
    let v: serde_json::Value =
        serde_json::from_str(slice).context("judge returned invalid JSON")?;
    let num = |key: &str| v.get(key).and_then(|x| x.as_f64()).unwrap_or(0.0).clamp(0.0, 1.0);
    Ok(JudgeScore {
        faithfulness: num("faithfulness"),
        citation_accuracy: num("citation_accuracy"),
        correctness: num("correctness"),
        refusal_correct: v.get("refusal_correct").and_then(|x| x.as_bool()),
        rationale: v
            .get("rationale")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
    })
}

// ---------------------------------------------------------------------------
// Прогон
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub ks: Vec<usize>,
    pub rerank: bool,
    pub judge: bool,
    pub context_k: usize,
    pub limit: Option<usize>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            ks: DEFAULT_KS.to_vec(),
            rerank: true,
            judge: false,
            context_k: 6,
            limit: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ItemResult {
    pub id: String,
    pub lang: String,
    pub topic: Option<String>,
    pub answerable: bool,
    pub question: String,
    pub retrieval: RetrievalScore,
    pub latency_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge: Option<JudgeScore>,
    /// топ-1 источник/локация — для разбора ошибок
    pub top1: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct AnswerAggregate {
    pub n: usize,
    pub faithfulness: f64,
    pub citation_accuracy: f64,
    pub correctness: f64,
    pub refusals_correct: usize,
    pub refusals_total: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigResult {
    pub label: String,
    pub store: String,
    pub embedding_model: String,
    pub rerank: bool,
    pub rerank_model: Option<String>,
    pub chunks: i64,
    pub retrieval: Aggregate,
    pub by_lang: BTreeMap<String, Aggregate>,
    pub by_topic: BTreeMap<String, Aggregate>,
    pub latency_ms: Latency,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answers: Option<AnswerAggregate>,
    pub items: Vec<ItemResult>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Latency {
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub max_ms: f64,
}

fn latency(mut xs: Vec<f64>) -> Latency {
    if xs.is_empty() {
        return Latency::default();
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pick = |q: f64| xs[((xs.len() - 1) as f64 * q).round() as usize];
    Latency {
        mean_ms: xs.iter().sum::<f64>() / xs.len() as f64,
        p50_ms: pick(0.5),
        p95_ms: pick(0.95),
        max_ms: *xs.last().unwrap(),
    }
}

fn describe_top1(r: &Retrieved) -> String {
    let c = &r.chunk;
    let mut parts = vec![c.source.clone()];
    if let Some(p) = c.page {
        parts.push(format!("page {p}"));
    }
    if let Some(s) = &c.section {
        parts.push(s.chars().take(60).collect());
    }
    if let Some(sh) = &c.sheet {
        parts.push(sh.clone());
    }
    if let Some(row) = c.row {
        parts.push(format!("row {row}"));
    }
    parts.join(" | ")
}

/// Прогоняет «золотой» набор по одному индексу.
pub async fn evaluate_store(
    cfg: &Config,
    store_path: &Path,
    label: &str,
    gold: &[GoldItem],
    opts: &RunOptions,
    client: Option<&DeepSeekClient>,
) -> Result<ConfigResult> {
    let store = Store::open(store_path, false)?;
    let chunks = store.load_all().with_context(|| {
        format!("cannot load index {}", store_path.display())
    })?;
    if chunks.is_empty() {
        anyhow::bail!("index {} is empty", store_path.display());
    }

    // Модель эмбеддинга берём из метаданных индекса: векторы в нём посчитаны
    // именно ею, и подменять модель при оценке нельзя.
    let mut cfg = cfg.clone();
    if let Some(m) = store.get_meta("embedding_model")? {
        cfg.embedding.model = m;
    }
    if let Some(l) = store.get_meta("embedding_max_length")? {
        if let Ok(n) = l.parse::<usize>() {
            cfg.embedding.max_length = Some(n);
        }
    }
    let cfg = &cfg;

    let mut embedder = Embedder::from_config(cfg)?;
    let mut reranker = if opts.rerank && cfg.rerank.enabled {
        Some(Reranker::new(
            &cfg.rerank.model,
            &cfg.models_dir,
            true,
            cfg.rerank.intra_threads,
            cfg.rerank.max_length,
        )?)
    } else {
        None
    };

    let max_k = opts.ks.iter().copied().max().unwrap_or(10);
    let items: Vec<&GoldItem> = match opts.limit {
        Some(n) => gold.iter().take(n).collect(),
        None => gold.iter().collect(),
    };

    // прогрев: первый запрос включает ленивую инициализацию ONNX-сессий
    let _ = pipeline::rank(
        &chunks,
        &embedder.embed_query("warmup query")?,
        "warmup query",
        &cfg.rerank,
        1,
        reranker.as_mut(),
    )?;

    let mut results: Vec<ItemResult> = Vec::with_capacity(items.len());
    let mut lat: Vec<f64> = Vec::with_capacity(items.len());
    let mut accum = Accum::new(&opts.ks);
    let mut by_lang: BTreeMap<String, Accum> = BTreeMap::new();
    let mut by_topic: BTreeMap<String, Accum> = BTreeMap::new();
    let mut ans = AnswerAggregate::default();

    for (i, item) in items.iter().enumerate() {
        let t0 = Instant::now();
        let qv = embedder.embed_query(&item.question)?;
        let retrieved = pipeline::rank(
            &chunks,
            &qv,
            &item.question,
            &cfg.rerank,
            max_k,
            reranker.as_mut(),
        )?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        lat.push(dt);

        let retrieval = score_retrieval(item, &retrieved, &opts.ks);
        if item.answerable {
            accum.push(&retrieval);
            by_lang
                .entry(item.lang.clone())
                .or_insert_with(|| Accum::new(&opts.ks))
                .push(&retrieval);
            by_topic
                .entry(item.topic.clone().unwrap_or_else(|| "other".into()))
                .or_insert_with(|| Accum::new(&opts.ks))
                .push(&retrieval);
        }

        let mut answer = None;
        let mut judge = None;
        if opts.judge {
            let client = client.context("--judge requires a DeepSeek client/API key")?;
            let ctx_items: Vec<Retrieved> = retrieved
                .iter()
                .take(opts.context_k)
                .cloned()
                .collect();
            let text = pipeline::generate(client, &item.question, &ctx_items).await?;
            let score = judge_answer(client, item, &ctx_items, &text).await?;
            ans.n += 1;
            ans.faithfulness += score.faithfulness;
            ans.citation_accuracy += score.citation_accuracy;
            ans.correctness += score.correctness;
            if !item.answerable {
                ans.refusals_total += 1;
                if score.refusal_correct.unwrap_or(false) {
                    ans.refusals_correct += 1;
                }
            }
            answer = Some(text);
            judge = Some(score);
        }

        eprintln!(
            "[eval] {label} {:>3}/{:<3} {:<28} RR={:.2} R@5={:.2} {:.0}ms",
            i + 1,
            items.len(),
            item.id.chars().take(28).collect::<String>(),
            retrieval.reciprocal_rank,
            retrieval.recall_at_k.get(&5).copied().unwrap_or(0.0),
            dt
        );

        results.push(ItemResult {
            id: item.id.clone(),
            lang: item.lang.clone(),
            topic: item.topic.clone(),
            answerable: item.answerable,
            question: item.question.clone(),
            retrieval,
            latency_ms: dt,
            answer,
            judge,
            top1: retrieved.first().map(describe_top1),
        });
    }

    if ans.n > 0 {
        let n = ans.n as f64;
        ans.faithfulness /= n;
        ans.citation_accuracy /= n;
        ans.correctness /= n;
    }

    Ok(ConfigResult {
        label: label.to_string(),
        store: store_path.display().to_string(),
        embedding_model: cfg.embedding.model.clone(),
        rerank: opts.rerank && cfg.rerank.enabled,
        rerank_model: if opts.rerank && cfg.rerank.enabled {
            Some(cfg.rerank.model.clone())
        } else {
            None
        },
        chunks: store.count()?,
        retrieval: accum.finish(),
        by_lang: by_lang.into_iter().map(|(k, v)| (k, v.finish())).collect(),
        by_topic: by_topic.into_iter().map(|(k, v)| (k, v.finish())).collect(),
        latency_ms: latency(lat),
        answers: if opts.judge { Some(ans) } else { None },
        items: results,
    })
}

async fn judge_answer(
    client: &DeepSeekClient,
    item: &GoldItem,
    retrieved: &[Retrieved],
    answer: &str,
) -> Result<JudgeScore> {
    let raw = client
        .chat_json(JUDGE_SYSTEM, &judge_prompt(item, retrieved, answer))
        .await?;
    parse_judge_json(&raw)
}

// ---------------------------------------------------------------------------
// Отчёт
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    pub generated_at: String,
    pub gold_items: usize,
    pub gold_files: Vec<String>,
    pub ks: Vec<usize>,
    pub store_path: String,
    pub configs: Vec<ConfigResult>,
}

/// Markdown-отчёт: сводка, метрики по k, задержки и разбивки по языку и теме.
pub fn render_markdown(report: &EvalReport) -> String {
    let mut md = String::new();
    md.push_str("# Оценка качества RAG (sdtmrag)\n\n");
    md.push_str(&format!("- Дата прогона: {}\n", report.generated_at));
    md.push_str(&format!(
        "- Вопросов в наборе: {} ({})\n",
        report.gold_items,
        report.gold_files.join(", ")
    ));
    md.push_str(&format!("- Индекс: `{}`\n", report.store_path));
    md.push_str(&format!("- k: {:?}\n\n", report.ks));

    // Заголовок таблицы «конфигурации в столбцах».
    let table_head = |first: &str| -> String {
        let mut h = format!("| {first} |");
        let mut s = String::from("|---|");
        for c in &report.configs {
            h.push_str(&format!(" {} |", c.label));
            s.push_str("---:|");
        }
        format!("{h}\n{s}\n")
    };
    md.push_str("## Сводка\n\n");
    md.push_str("| Конфигурация | Модель эмбеддингов | Реранк | Чанков | MRR | Задержка p50, мс |\n");
    md.push_str("|---|---|---|---:|---:|---:|\n");
    for c in &report.configs {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {:.3} | {:.0} |\n",
            c.label,
            c.embedding_model,
            c.rerank_model.as_deref().unwrap_or("выключен"),
            c.chunks,
            c.retrieval.mrr,
            c.latency_ms.p50_ms
        ));
    }

    for metric in ["precision", "recall", "ndcg", "hit"] {
        let title = match metric {
            "precision" => "Precision@k",
            "recall" => "Recall@k",
            "ndcg" => "nDCG@k",
            _ => "Hit@k",
        };
        md.push_str(&format!("\n### {title}\n\n"));
        md.push_str(&table_head("k"));
        for &k in &report.ks {
            md.push_str(&format!("| {k} |"));
            for c in &report.configs {
                let m = match metric {
                    "precision" => &c.retrieval.precision_at_k,
                    "recall" => &c.retrieval.recall_at_k,
                    "ndcg" => &c.retrieval.ndcg_at_k,
                    _ => &c.retrieval.hit_at_k,
                };
                md.push_str(&format!(" {:.3} |", m.get(&k).copied().unwrap_or(0.0)));
            }
            md.push('\n');
        }
    }

    md.push_str("\n### Задержка (мс)\n\n");
    md.push_str("| Конфигурация | mean | p50 | p95 | max |\n|---|---:|---:|---:|---:|\n");
    for c in &report.configs {
        md.push_str(&format!(
            "| {} | {:.0} | {:.0} | {:.0} | {:.0} |\n",
            c.label,
            c.latency_ms.mean_ms,
            c.latency_ms.p50_ms,
            c.latency_ms.p95_ms,
            c.latency_ms.max_ms
        ));
    }

    if report.configs.iter().any(|c| c.answers.is_some()) {
        md.push_str("\n### Качество ответов (LLM-судья, 0..1)\n\n");
        md.push_str(
            "| Конфигурация | Faithfulness | Citation accuracy | Correctness | Верных отказов |\n|---|---:|---:|---:|---:|\n",
        );
        for c in &report.configs {
            if let Some(a) = &c.answers {
                md.push_str(&format!(
                    "| {} | {:.3} | {:.3} | {:.3} | {}/{} |\n",
                    c.label, a.faithfulness, a.citation_accuracy, a.correctness,
                    a.refusals_correct, a.refusals_total
                ));
            }
        }
    }

    md.push_str("\n## Разбивка по языку вопроса (Recall@5)\n\n");
    md.push_str(&table_head("Язык"));
    let langs: Vec<String> = report
        .configs
        .iter()
        .flat_map(|c| c.by_lang.keys().cloned())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    for lang in &langs {
        md.push_str(&format!("| {lang} |"));
        for c in &report.configs {
            match c.by_lang.get(lang) {
                Some(a) => md.push_str(&format!(
                    " {:.3} (n={}) |",
                    a.recall_at_k.get(&5).copied().unwrap_or(0.0),
                    a.n
                )),
                None => md.push_str(" — |"),
            }
        }
        md.push('\n');
    }

    md.push_str("\n## Разбивка по темам (Recall@5)\n\n");
    md.push_str(&table_head("Тема"));
    let topics: Vec<String> = report
        .configs
        .iter()
        .flat_map(|c| c.by_topic.keys().cloned())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    for topic in &topics {
        md.push_str(&format!("| {topic} |"));
        for c in &report.configs {
            match c.by_topic.get(topic) {
                Some(a) => md.push_str(&format!(
                    " {:.3} (n={}) |",
                    a.recall_at_k.get(&5).copied().unwrap_or(0.0),
                    a.n
                )),
                None => md.push_str(" — |"),
            }
        }
        md.push('\n');
    }

    md.push_str("\n## MRR по темам\n\n");
    md.push_str(&table_head("Тема"));
    for topic in &topics {
        md.push_str(&format!("| {topic} |"));
        for c in &report.configs {
            match c.by_topic.get(topic) {
                Some(a) => md.push_str(&format!(" {:.3} |", a.mrr)),
                None => md.push_str(" — |"),
            }
        }
        md.push('\n');
    }

    md.push_str(&format!(
        "\n_Столбцы во всех таблицах: {}._\n",
        report
            .configs
            .iter()
            .map(|c| c.label.clone())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    md
}

/// Пишет `report.json` и `report.md` в каталог.
pub fn write_report(report: &EvalReport, out_dir: &Path) -> Result<(PathBuf, PathBuf)> {
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("cannot create report dir: {}", out_dir.display()))?;
    let json_path = out_dir.join("report.json");
    std::fs::write(&json_path, serde_json::to_string_pretty(report)?)
        .with_context(|| format!("cannot write {}", json_path.display()))?;
    let md_path = out_dir.join("report.md");
    std::fs::write(&md_path, render_markdown(report))
        .with_context(|| format!("cannot write {}", md_path.display()))?;
    Ok((json_path, md_path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::StoredChunk;

    fn chunk(source: &str, page: i64, section: &str, text: &str) -> StoredChunk {
        StoredChunk {
            id: 1,
            source: source.to_string(),
            kind: "pdf".to_string(),
            page: Some(page),
            section: Some(section.to_string()),
            sheet: None,
            row: None,
            text: text.to_string(),
            embedding: vec![0.0; 4],
        }
    }

    fn item() -> GoldItem {
        GoldItem {
            id: "t1".into(),
            question: "q".into(),
            lang: "en".into(),
            topic: Some("LB".into()),
            answerable: true,
            expected: vec![Locator {
                source: "doc.pdf".into(),
                page: Some(10),
                section: None,
                sheet: None,
                row: None,
                contains: None,
            }],
            reference_answer: None,
            needs_review: false,
            notes: None,
        }
    }

    #[test]
    fn locator_matches_by_source_and_page() {
        let l = Locator {
            source: "doc.pdf".into(),
            page: Some(10),
            section: None,
            sheet: None,
            row: None,
            contains: None,
        };
        assert!(l.matches(&chunk("doc.pdf", 10, "5 X", "text")));
        assert!(!l.matches(&chunk("doc.pdf", 11, "5 X", "text")));
        assert!(!l.matches(&chunk("other.pdf", 10, "5 X", "text")));
    }

    #[test]
    fn locator_section_is_case_and_space_insensitive() {
        let l = Locator {
            source: "doc.pdf".into(),
            page: None,
            section: Some("6.3.5  Laboratory Test Results".into()),
            sheet: None,
            row: None,
            contains: None,
        };
        assert!(l.matches(&chunk("doc.pdf", 1, "6.3.5 Laboratory Test Results", "t")));
    }

    #[test]
    fn metrics_reward_early_hits() {
        let it = item();
        let mk = |page: i64| Retrieved {
            score: 1.0,
            chunk: chunk("doc.pdf", page, "s", "text"),
        };
        let good = vec![mk(10), mk(1), mk(2)];
        let bad = vec![mk(1), mk(2), mk(10)];
        let g = score_retrieval(&it, &good, &[1, 3]);
        let b = score_retrieval(&it, &bad, &[1, 3]);
        assert_eq!(g.reciprocal_rank, 1.0);
        assert_eq!(b.reciprocal_rank, 1.0 / 3.0);
        assert!(g.ndcg_at_k[&3] > b.ndcg_at_k[&3]);
        assert_eq!(g.hit_at_k[&1], 1.0);
        assert_eq!(b.hit_at_k[&1], 0.0);
    }

    #[test]
    fn parses_judge_json_with_fences() {
        let raw = "```json\n{\"faithfulness\": 0.8, \"citation_accuracy\": 1, \
                   \"correctness\": 0.5, \"refusal_correct\": null, \"rationale\": \"ok\"}\n```";
        let s = parse_judge_json(raw).unwrap();
        assert!((s.faithfulness - 0.8).abs() < 1e-9);
        assert_eq!(s.citation_accuracy, 1.0);
        assert!(s.refusal_correct.is_none());
    }
}
