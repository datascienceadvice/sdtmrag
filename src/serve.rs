use crate::config::Config;
use crate::embed::Reranker;
use crate::generation::DeepSeekClient;
use crate::pipeline;
use crate::store::Store;
use anyhow::Result;
use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

pub struct AppState {
    pub cfg: Config,
    pub store: Mutex<Store>,
    pub embedder: Mutex<crate::embed::Embedder>,
    pub reranker: Mutex<Option<Reranker>>,
    pub client: DeepSeekClient,
}

type Shared = Arc<AppState>;

#[derive(Deserialize)]
pub struct QueryRequest {
    pub question: String,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default = "default_true")]
    pub rerank: bool,
    #[serde(default = "default_true")]
    pub generate: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub answer: Option<String>,
    pub model: String,
    pub sources: Vec<SourceHit>,
}

#[derive(Serialize)]
pub struct SourceHit {
    pub score: f32,
    pub source: String,
    pub kind: String,
    pub page: Option<i64>,
    pub section: Option<String>,
    pub sheet: Option<String>,
    pub row: Option<i64>,
    pub preview: String,
}

async fn health(State(state): State<Shared>) -> Json<serde_json::Value> {
    let chunks = state.store.lock().unwrap().count().unwrap_or(0);
    let model = state.embedder.lock().unwrap().model_name().to_string();
    Json(serde_json::json!({
        "status": "ok",
        "chunks": chunks,
        "embedding_model": model,
    }))
}

async fn stats(State(state): State<Shared>) -> Json<serde_json::Value> {
    let store = state.store.lock().unwrap();
    let model = state.embedder.lock().unwrap().model_name().to_string();
    Json(serde_json::json!({
        "chunks": store.count().unwrap_or(0),
        "sources": store.sources().unwrap_or_default(),
        "embedding_model": model,
        "deepseek_model": state.client.model(),
        "rerank_enabled": state.cfg.rerank.enabled,
        "rerank_model": state.cfg.rerank.model,
    }))
}

async fn documents(State(state): State<Shared>) -> Json<serde_json::Value> {
    let store = state.store.lock().unwrap();
    Json(serde_json::json!({ "sources": store.sources().unwrap_or_default() }))
}

async fn query(State(state): State<Shared>, Json(req): Json<QueryRequest>) -> Result<Json<QueryResponse>, (StatusCode, String)> {
    if req.question.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "question is required".into()));
    }

    // Синхронный поиск: блокировки отпускаем до первого await (иначе future не Send).
    let final_k = req
        .top_k
        .unwrap_or(state.cfg.rerank.final_top_k)
        .max(state.cfg.rerank.final_top_k);
    let retrieved = {
        let store = state.store.lock().unwrap();
        let mut embedder = state.embedder.lock().unwrap();
        let result = if req.rerank && state.cfg.rerank.enabled {
            let mut slot = state.reranker.lock().unwrap();
            if slot.is_none() {
                *slot = Some(Reranker::new(
                    &state.cfg.rerank.model,
                    &state.cfg.models_dir,
                    true,
                    state.cfg.rerank.intra_threads,
                    state.cfg.rerank.max_length,
                )
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("reranker init: {e}")))?);
            }
            pipeline::search(
                &store,
                &mut embedder,
                &req.question,
                &state.cfg.rerank,
                final_k,
                slot.as_mut(),
            )
        } else {
            pipeline::search(
                &store,
                &mut embedder,
                &req.question,
                &state.cfg.rerank,
                final_k,
                None,
            )
        };
        match result {
            Ok(v) => v,
            Err(e) => return Err((StatusCode::INTERNAL_SERVER_ERROR, format!("search failed: {e:#}"))),
        }
    };

    let mut hits: Vec<SourceHit> = retrieved.iter().map(to_hit).collect();
    if let Some(k) = req.top_k {
        hits.truncate(k);
    }

    let answer = if req.generate {
        match pipeline::generate(&state.client, &req.question, &retrieved).await {
            Ok(a) => Some(a),
            Err(e) => return Err((StatusCode::INTERNAL_SERVER_ERROR, format!("generation failed: {e:#}"))),
        }
    } else {
        None
    };

    Ok(Json(QueryResponse {
        answer,
        model: state.client.model().to_string(),
        sources: hits,
    }))
}

fn to_hit(r: &pipeline::Retrieved) -> SourceHit {
    SourceHit {
        score: r.score,
        source: r.chunk.source.clone(),
        kind: r.chunk.kind.clone(),
        page: r.chunk.page,
        section: r.chunk.section.clone(),
        sheet: r.chunk.sheet.clone(),
        row: r.chunk.row,
        preview: r.chunk.text.chars().take(600).collect(),
    }
}

pub fn build_router(state: Shared) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/stats", get(stats))
        .route("/api/documents", get(documents))
        .route("/api/query", post(query))
        .layer(DefaultBodyLimit::max(1024 * 1024))
        .with_state(state)
}

pub async fn serve(cfg: Config, store: Store, client: DeepSeekClient) -> Result<()> {
    // Модель эмбеддингов — из метаданных индекса, а не из конфигурации.
    let mut cfg = cfg;
    store.apply_embedding_meta(&mut cfg)?;
    let embedder = crate::embed::Embedder::from_config(&cfg)?;
let state: Shared = Arc::new(AppState {
        cfg: cfg.clone(),
        store: Mutex::new(store),
        embedder: Mutex::new(embedder),
        reranker: Mutex::new(None),
        client,
    });
    let app = build_router(state);
    let addr = format!("{}:{}", cfg.server.bind, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("sdtmrag server listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}