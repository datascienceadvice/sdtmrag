//! Лексический baseline: BM25 и слияние ранжирований (RRF).
//!
//! Нужен, чтобы сравнение «очистка против отсутствия очистки» не сводилось к
//! сравнению системы с самой собой: рецензент вправе спросить, не решает ли
//! задачу обычный поиск по ключевым словам. BM25 считается по тем же чанкам,
//! что и плотный поиск, поэтому обе ветви используют один корпус, одно
//! разбиение и одни локаторы.

use crate::pipeline::Retrieved;
use crate::store::StoredChunk;
use std::collections::HashMap;

/// Токенизация: алфавитно-цифровые последовательности, нижний регистр.
/// Стемминг намеренно не применяется — иначе baseline был бы сильнее, чем
/// описано в статье, и сравнение перестало бы быть воспроизводимым.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            cur.extend(ch.to_lowercase());
        } else if !cur.is_empty() {
            if cur.chars().count() >= 2 {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if cur.chars().count() >= 2 {
        out.push(cur);
    }
    out
}

pub struct Bm25Index {
    /// термин -> [(номер документа, частота)]
    postings: HashMap<String, Vec<(usize, u32)>>,
    idf: HashMap<String, f32>,
    lens: Vec<f32>,
    avgdl: f32,
    k1: f32,
    b: f32,
}

impl Bm25Index {
    pub fn build(all: &[StoredChunk]) -> Self {
        let mut postings: HashMap<String, Vec<(usize, u32)>> = HashMap::new();
        let mut lens = Vec::with_capacity(all.len());
        let mut total = 0usize;
        for (i, chunk) in all.iter().enumerate() {
            let tokens = tokenize(&chunk.text);
            total += tokens.len();
            lens.push(tokens.len() as f32);
            let mut tf: HashMap<String, u32> = HashMap::new();
            for t in tokens {
                *tf.entry(t).or_insert(0) += 1;
            }
            for (term, n) in tf {
                postings.entry(term).or_default().push((i, n));
            }
        }
        let n = all.len().max(1) as f32;
        let idf = postings
            .iter()
            .map(|(term, docs)| {
                let df = docs.len() as f32;
                (term.clone(), (1.0 + (n - df + 0.5) / (df + 0.5)).ln())
            })
            .collect();
        Self {
            avgdl: if all.is_empty() { 0.0 } else { total as f32 / all.len() as f32 },
            postings,
            idf,
            lens,
            k1: 1.2,
            b: 0.75,
        }
    }

    /// Возвращает (номер чанка, BM25-оценка), по убыванию оценки.
    pub fn search(&self, question: &str, k: usize) -> Vec<(usize, f32)> {
        let query = tokenize(question);
        if query.is_empty() {
            return Vec::new();
        }
        let mut scores = vec![0.0f32; self.lens.len()];
        for term in &query {
            let (Some(idf), Some(docs)) = (self.idf.get(term), self.postings.get(term)) else {
                continue;
            };
            for &(doc, tf) in docs {
                let dl = self.lens[doc];
                let denom = tf as f32 + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl);
                scores[doc] += idf * (tf as f32 * (self.k1 + 1.0)) / denom;
            }
        }
        let mut ranked: Vec<(usize, f32)> = scores
            .into_iter()
            .enumerate()
            .filter(|(_, s)| *s > 0.0)
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);
        ranked
    }

    pub fn retrieved(&self, all: &[StoredChunk], question: &str, k: usize) -> Vec<Retrieved> {
        self.search(question, k)
            .into_iter()
            .filter_map(|(i, score)| {
                all.get(i).map(|c| Retrieved {
                    score,
                    chunk: c.clone(),
                })
            })
            .collect()
    }
}

/// Reciprocal rank fusion: устойчивый способ объединить два ранжирования без
/// подбора весов (k = 60 — стандартное значение из литературы).
pub fn rrf(a: &[Retrieved], b: &[Retrieved], limit: usize) -> Vec<Retrieved> {
    const K: f32 = 60.0;
    let mut score: HashMap<i64, f32> = HashMap::new();
    let mut keep: HashMap<i64, Retrieved> = HashMap::new();
    for list in [a, b] {
        for (rank, item) in list.iter().enumerate() {
            *score.entry(item.chunk.id).or_insert(0.0) += 1.0 / (K + rank as f32 + 1.0);
            keep.entry(item.chunk.id).or_insert_with(|| item.clone());
        }
    }
    let mut merged: Vec<(i64, f32)> = score.into_iter().collect();
    merged.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    merged
        .into_iter()
        .take(limit)
        .filter_map(|(id, s)| {
            keep.remove(&id).map(|mut r| {
                r.score = s;
                r
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::StoredChunk;

    fn chunk(id: i64, text: &str) -> StoredChunk {
        StoredChunk {
            id,
            source: "doc.pdf".into(),
            kind: "pdf".into(),
            page: Some(1),
            section: None,
            sheet: None,
            row: None,
            text: text.into(),
            embedding: vec![0.0; 4],
        }
    }

    #[test]
    fn tokenizer_is_case_and_punctuation_insensitive() {
        assert_eq!(tokenize("LBTESTCD (Lab Test), g/L"), vec!["lbtestcd", "lab", "test"]);
        assert!(tokenize("a - b").is_empty(), "однобуквенные токены отбрасываются");
    }

    #[test]
    fn exact_term_wins_over_unrelated_document() {
        let all = vec![
            chunk(1, "The variable LBORRES holds the original result in original units."),
            chunk(2, "Microbiology susceptibility testing reports minimum inhibitory concentrations."),
            chunk(3, "Laboratory test results are described by the LB domain, including reference ranges."),
        ];
        let idx = Bm25Index::build(&all);
        let top = idx.search("LBORRES", 2);
        assert_eq!(top[0].0, 0, "документ с точным термином должен быть первым");
        assert!(top[0].1 > 0.0);
    }

    #[test]
    fn rrf_promotes_documents_found_by_both_rankers() {
        let all = vec![chunk(1, "one"), chunk(2, "two"), chunk(3, "three")];
        let a = vec![
            Retrieved { score: 0.9, chunk: all[0].clone() },
            Retrieved { score: 0.5, chunk: all[1].clone() },
        ];
        let b = vec![
            Retrieved { score: 7.0, chunk: all[1].clone() },
            Retrieved { score: 3.0, chunk: all[2].clone() },
        ];
        let merged = rrf(&a, &b, 3);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].chunk.id, 2, "документ, найденный обоими, идёт первым");
    }
}
