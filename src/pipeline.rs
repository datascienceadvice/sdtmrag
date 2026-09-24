use crate::config::RerankConfig;
use crate::embed::{Embedder, Reranker};
use crate::generation::DeepSeekClient;
use crate::store::{StoredChunk, Store};
use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct Retrieved {
    pub score: f32,
    pub chunk: StoredChunk,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
    }
    dot
}

/// Dense-поиск + опциональный реранк. Векторы уже нормализованы -> cos = dot.
///
/// `final_k` — сколько результатов вернуть после реранка/усечения.
pub fn search(
    store: &Store,
    embedder: &mut Embedder,
    question: &str,
    cfg: &RerankConfig,
    final_k: usize,
    reranker: Option<&mut Reranker>,
) -> Result<Vec<Retrieved>> {
    let q = embedder.embed_query(question)?;
    let all = store.load_all()?;
    rank(&all, &q, question, cfg, final_k, reranker)
}

/// Ранжирование по уже загруженному корпусу.
///
/// Позволяет не перечитывать весь индекс из SQLite на каждый запрос
/// (в оценке качества это сотни запросов подряд).
pub fn rank(
    all: &[StoredChunk],
    query_vec: &[f32],
    question: &str,
    cfg: &RerankConfig,
    final_k: usize,
    reranker: Option<&mut Reranker>,
) -> Result<Vec<Retrieved>> {
    if all.is_empty() {
        return Ok(Vec::new());
    }

    // Вектор запроса должен быть посчитан той же моделью, что и векторы индекса.
    // Если размерности разошлись, поиск вернул бы случайный порядок — лучше
    // отказать явно.
    let dim = all[0].embedding.len();
    if query_vec.len() != dim {
        anyhow::bail!(
            "embedding dimension mismatch: index stores {dim}-dimensional vectors, \
             but the query was embedded into {} dimensions — the query model differs \
             from the model the index was built with",
            query_vec.len()
        );
    }

    // Сортируем индексы, а не копии чанков: при 3-6 тыс. чанков клонирование
    // всего корпуса на каждый запрос заметно дороже самого перебора.
    let mut scored: Vec<(f32, usize)> = all
        .iter()
        .enumerate()
        .map(|(i, c)| (cosine(query_vec, &c.embedding), i))
        .filter(|(s, _)| s.is_finite())
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(cfg.candidates);

    if let Some(rr) = reranker {
        if cfg.enabled && !scored.is_empty() {
            let texts: Vec<String> = scored.iter().map(|(_, i)| all[*i].text.clone()).collect();
            let docs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let order = rr.rerank(question, &docs).context("rerank failed")?;
            let n = order.len().min(final_k) as f32;
            let mut out = Vec::with_capacity(final_k);
            for (pos, i) in order.into_iter().take(final_k).enumerate() {
                if let Some((_, idx)) = scored.get(i) {
                    out.push(Retrieved {
                        // позиционная метрика реранка: 1.0 для лучшего, далее по убыванию
                        score: if n > 1.0 { 1.0 - pos as f32 / n } else { 1.0 },
                        chunk: all[*idx].clone(),
                    });
                }
            }
            return Ok(out);
        }
    }

    Ok(scored
        .into_iter()
        .take(final_k)
        .map(|(score, idx)| Retrieved {
            score,
            chunk: all[idx].clone(),
        })
        .collect())
}

/// Превращает найденные чанки в текст контекста с источниками.
pub fn render_context(retrieved: &[Retrieved]) -> String {
    let mut context = String::new();
    for (i, r) in retrieved.iter().enumerate() {
        let c = &r.chunk;
        context.push_str(&format!(
            "--- Excerpt {} [source: {}]{}{}{}{} ---\n{}\n\n",
            i + 1,
            c.source,
            c.page.map(|p| format!(", page {p}")).unwrap_or_default(),
            c.section
                .as_ref()
                .map(|s| format!(", section: {s}"))
                .unwrap_or_default(),
            c.sheet
                .as_ref()
                .map(|s| format!(", sheet: {s}"))
                .unwrap_or_default(),
            c.row.map(|r| format!(", row {r}")).unwrap_or_default(),
            c.text,
        ));
    }
    context
}

pub fn system_prompt() -> &'static str {
    "You are an expert assistant on the CDISC SDTM standard (Study Data Tabulation Model) \
     and the SDTM Implementation Guide (SDTMIG). \
     Answer the user's question using ONLY the documentation excerpts provided. \
     Cite the source of each statement using the excerpt markers ([1], [2], ...). \
     If the excerpts do not contain the information needed to answer, state clearly that \
     the information was not found in the provided documentation. \
     Be precise, concise and technical. Do not invent variable names, domains or rules."
}

pub async fn generate(
    client: &DeepSeekClient,
    question: &str,
    retrieved: &[Retrieved],
) -> Result<String> {
    let context = render_context(retrieved);
    let user = format!("Question: {question}\n\nDocumentation excerpts:\n{context}");
    client.chat(system_prompt(), &user).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::StoredChunk;

    fn chunk(id: i64, embedding: Vec<f32>) -> StoredChunk {
        StoredChunk {
            id,
            source: "doc.pdf".into(),
            kind: "pdf".into(),
            page: Some(1),
            section: None,
            sheet: None,
            row: None,
            text: format!("chunk {id}"),
            embedding,
        }
    }

    #[test]
    fn rejects_embedding_dimension_mismatch() {
        // Индекс собран 4-мерной моделью, запрос — 3-мерной: поиск должен
        // отказать, а не вернуть случайный порядок.
        let all = vec![chunk(1, vec![1.0, 0.0, 0.0, 0.0])];
        let err = rank(&all, &[1.0, 0.0, 0.0], "q", &RerankConfig::default(), 3, None)
            .unwrap_err();
        assert!(
            err.to_string().contains("dimension mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ranks_closest_chunk_first() {
        let all = vec![
            chunk(1, vec![1.0, 0.0]),
            chunk(2, vec![0.0, 1.0]),
            chunk(3, vec![0.9, 0.1]),
        ];
        let out = rank(&all, &[1.0, 0.0], "q", &RerankConfig::default(), 2, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].chunk.id, 1);
        assert_eq!(out[1].chunk.id, 3);
        assert!(out[0].score >= out[1].score);
    }
}