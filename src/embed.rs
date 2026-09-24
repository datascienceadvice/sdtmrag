use anyhow::{Context, Result};
use fastembed::{
    EmbeddingModel, InitOptions, RerankInitOptions, RerankerModel, TextEmbedding, TextRerank,
};
use std::path::Path;

pub fn embedding_model(name: &str) -> Result<EmbeddingModel> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "bge-small-en-v1.5" | "bge-small" => EmbeddingModel::BGESmallENV15,
        "bge-base-en-v1.5" | "bge-base" => EmbeddingModel::BGEBaseENV15,
        "bge-large-en-v1.5" | "bge-large" => EmbeddingModel::BGELargeENV15,
        "bge-m3" => EmbeddingModel::BGEM3,
        "all-minilm-l6-v2" | "minilm" => EmbeddingModel::AllMiniLML6V2,
        "all-minilm-l12-v2" => EmbeddingModel::AllMiniLML12V2,
        "all-mpnet-base-v2" => EmbeddingModel::AllMpnetBaseV2,
        // многоязычные модели — нужны, когда запросы и документы на разных языках
        "multilingual-e5-small" | "e5-small" => EmbeddingModel::MultilingualE5Small,
        "multilingual-e5-base" | "e5-base" => EmbeddingModel::MultilingualE5Base,
        "multilingual-e5-large" | "e5-large" => EmbeddingModel::MultilingualE5Large,
        "paraphrase-multilingual-minilm-l12-v2" | "paraphrase-ml-minilm" => {
            EmbeddingModel::ParaphraseMLMiniLML12V2
        }
        "paraphrase-multilingual-mpnet-base-v2" | "paraphrase-ml-mpnet" => {
            EmbeddingModel::ParaphraseMLMpnetBaseV2
        }
        other => anyhow::bail!(
            "unsupported embedding model: {other} (bge-small-en-v1.5, bge-m3, \
             multilingual-e5-small, all-minilm-l6-v2, ...)"
        ),
    })
}

pub fn reranker_model(name: &str) -> Result<RerankerModel> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "bge-reranker-base" => RerankerModel::BGERerankerBase,
        "bge-reranker-v2-m3" => RerankerModel::BGERerankerV2M3,
        "jina-reranker-v1-turbo-en" => RerankerModel::JINARerankerV1TurboEn,
        "jina-reranker-v2-base-multilingual" => RerankerModel::JINARerankerV2BaseMultiligual,
        other => anyhow::bail!(
            "unsupported reranker model: {other} (bge-reranker-base, bge-reranker-v2-m3, jina-reranker-v1-turbo-en)"
        ),
    })
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Префиксы запроса и документа, с которыми модель обучалась.
///
/// bge-модели без инструкции заметно хуже ищут, а семейство E5 обучено на парах
/// `query: …` / `passage: …` — без этих префиксов качество поиска падает, поэтому
/// они обязательны, если мы хотим честно сравнивать модели между собой.
fn instruction_prefixes(model: &str) -> (&'static str, &'static str) {
    let name = model.to_ascii_lowercase();
    if name.contains("e5") {
        ("query: ", "passage: ")
    } else if name.contains("bge") {
        ("Represent this sentence for searching relevant passages: ", "")
    } else {
        ("", "")
    }
}

pub struct Embedder {
    model: TextEmbedding,
    model_name: String,
    query_prefix: &'static str,
    doc_prefix: &'static str,
}

impl Embedder {
    pub fn from_config(cfg: &crate::config::Config) -> Result<Self> {
        Self::new(
            &cfg.embedding.model,
            &cfg.models_dir,
            true,
            cfg.embedding.max_length,
            cfg.embedding.intra_threads,
        )
    }

    pub fn new(
        model_name: &str,
        cache_dir: &Path,
        show_download_progress: bool,
        max_length: Option<usize>,
        intra_threads: usize,
    ) -> Result<Self> {
        let model = embedding_model(model_name)?;
        let mut opts = InitOptions::new(model)
            .with_cache_dir(cache_dir.to_path_buf())
            .with_show_download_progress(show_download_progress);
        if let Some(l) = max_length {
            opts = opts.with_max_length(l);
        }
        if intra_threads > 0 {
            opts = opts.with_intra_threads(intra_threads);
        }
        let model = TextEmbedding::try_new(opts).context("failed to init TextEmbedding")?;
        let (query_prefix, doc_prefix) = instruction_prefixes(model_name);
        Ok(Self {
            model,
            model_name: model_name.to_string(),
            query_prefix,
            doc_prefix,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let inputs: Vec<String> = if self.doc_prefix.is_empty() {
            texts.to_vec()
        } else {
            texts
                .iter()
                .map(|t| format!("{}{}", self.doc_prefix, t))
                .collect()
        };
        self.model
            .embed(inputs, None)
            .context("embedding failed")
            .map(|v| v.into_iter().map(|e| normalize(&e)).collect())
    }

    pub fn embed_query(&mut self, question: &str) -> Result<Vec<f32>> {
        let input = format!("{}{}", self.query_prefix, question);
        let v = self
            .model
            .embed(vec![input], None)
            .context("query embedding failed")?;
        Ok(normalize(&v[0]))
    }
}

pub struct Reranker {
    model: TextRerank,
    model_name: String,
    max_length: usize,
}

impl Reranker {
    pub fn new(
        model_name: &str,
        cache_dir: &Path,
        show_download_progress: bool,
        intra_threads: usize,
        max_length: usize,
    ) -> Result<Self> {
        let model = reranker_model(model_name)?;
        let mut opts = RerankInitOptions::new(model)
            .with_cache_dir(cache_dir.to_path_buf())
            .with_show_download_progress(show_download_progress);
        if intra_threads > 0 {
            opts = opts.with_intra_threads(intra_threads);
        }
        if max_length > 0 {
            opts = opts.with_max_length(max_length);
        }
        let model = TextRerank::try_new(opts).context("failed to init TextRerank")?;
        Ok(Self {
            model,
            model_name: model_name.to_string(),
            max_length,
        })
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Возвращает индексы документов, отсортированные по убыванию релевантности.
    pub fn rerank(&mut self, query: &str, documents: &[&str]) -> Result<Vec<usize>> {
        let results = self
            .model
            .rerank(query, documents, false, Some(64))
            .context("rerank failed")?;
        Ok(results.into_iter().map(|r| r.index).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_prefixes_follow_model_family() {
        // E5 обучен на парах query/passage: без префиксов качество поиска падает
        assert_eq!(
            instruction_prefixes("multilingual-e5-small"),
            ("query: ", "passage: ")
        );
        assert_eq!(
            instruction_prefixes("MultilingualE5Base"),
            ("query: ", "passage: ")
        );
        // bge нужна инструкция только к запросу
        let (q, d) = instruction_prefixes("bge-small-en-v1.5");
        assert!(q.starts_with("Represent this sentence"));
        assert_eq!(d, "");
        assert_eq!(instruction_prefixes("bge-m3"), (q, ""));
        // прочие модели — без префиксов
        assert_eq!(instruction_prefixes("all-minilm-l6-v2"), ("", ""));
    }

    #[test]
    fn embedding_model_accepts_multilingual_names() {
        assert!(embedding_model("multilingual-e5-small").is_ok());
        assert!(embedding_model("multilingual-e5-base").is_ok());
        assert!(embedding_model("paraphrase-multilingual-minilm-l12-v2").is_ok());
        assert!(embedding_model("bge-m3").is_ok());
        assert!(embedding_model("no-such-model").is_err());
    }
}