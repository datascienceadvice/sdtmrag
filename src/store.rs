use crate::chunk::ChunkRecord;
use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct StoredChunk {
    pub id: i64,
    pub source: String,
    pub kind: String,
    pub page: Option<i64>,
    pub section: Option<String>,
    pub sheet: Option<String>,
    pub row: Option<i64>,
    pub text: String,
    pub embedding: Vec<f32>,
}

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(path: &Path, rebuild: bool) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("cannot create dir: {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("cannot open sqlite store: {}", path.display()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        if rebuild {
            conn.execute_batch("DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS meta;")?;
        }
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                kind TEXT NOT NULL,
                page INTEGER,
                section TEXT,
                sheet TEXT,
                row INTEGER,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );",
        )?;
        Ok(Self { conn })
    }

    /// Записывает метаданные сборки индекса (модель, параметры чанкинга и
    /// очистки, статистику). Индекс становится самодостаточным: при оценке
    /// качества модель эмбеддинга берётся из него, а не из текущего конфига.
    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare("SELECT value FROM meta WHERE key = ?1")?;
        let mut rows = stmt.query(params![key])?;
        match rows.next()? {
            Some(r) => Ok(Some(r.get(0)?)),
            None => Ok(None),
        }
    }

    pub fn all_meta(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare("SELECT key, value FROM meta ORDER BY key")?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Переносит в конфигурацию модель эмбеддингов, которой был построен индекс.
    ///
    /// Без этого запрос к индексу, собранному другой моделью, молча сравнивал бы
    /// векторы разной размерности: `cosine` останавливается на более коротком
    /// векторе и возвращает бессмысленное число, а поиск деградирует до
    /// случайного. Индекс всегда знает свою модель — источником истины должен
    /// быть он, а не текущий config.toml.
    pub fn apply_embedding_meta(&self, cfg: &mut crate::config::Config) -> Result<()> {
        if let Some(model) = self.get_meta("embedding_model")? {
            if !model.is_empty() && model != cfg.embedding.model {
                eprintln!(
                    "[store] embedding model taken from the index: {} (config had {})",
                    model, cfg.embedding.model
                );
                cfg.embedding.model = model;
            }
        }
        if let Some(len) = self.get_meta("embedding_max_length")? {
            if let Ok(n) = len.parse::<usize>() {
                cfg.embedding.max_length = Some(n);
            }
        }
        Ok(())
    }

    pub fn insert_batch(&self, recs: &[(ChunkRecord, Vec<f32>)]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO chunks (source, kind, page, section, sheet, row, text, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?;
            for (rec, emb) in recs {
                let mut blob = Vec::with_capacity(emb.len() * 4);
                for f in emb {
                    blob.extend_from_slice(&f.to_le_bytes());
                }
                stmt.execute(params![
                    rec.source,
                    match rec.kind {
                        crate::chunk::ChunkKind::Pdf => "pdf",
                        crate::chunk::ChunkKind::Xlsx => "xlsx",
                    },
                    rec.page.map(|p| p as i64),
                    rec.section,
                    rec.sheet,
                    rec.row.map(|r| r as i64),
                    rec.text,
                    blob,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn count(&self) -> Result<i64> {
        let mut stmt = self.conn.prepare("SELECT COUNT(*) FROM chunks")?;
        Ok(stmt.query_row([], |r| r.get(0))?)
    }

    pub fn sources(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt =
            self.conn.prepare("SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source")?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn load_all(&self) -> Result<Vec<StoredChunk>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, source, kind, page, section, sheet, row, text, embedding FROM chunks",
        )?;
        let rows = stmt.query_map([], |r| {
            let embed_blob: Vec<u8> = r.get(8)?;
            let embedding = embed_blob
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(StoredChunk {
                id: r.get(0)?,
                source: r.get(1)?,
                kind: r.get(2)?,
                page: r.get(3)?,
                section: r.get(4)?,
                sheet: r.get(5)?,
                row: r.get(6)?,
                text: r.get(7)?,
                embedding,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }
}