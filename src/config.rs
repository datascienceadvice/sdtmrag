use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub inputs: PathBuf,
    pub data_dir: PathBuf,
    pub store_path: PathBuf,
    pub models_dir: PathBuf,
    pub embedding: EmbeddingConfig,
    pub chunking: ChunkingConfig,
    pub cleaning: CleaningConfig,
    pub rerank: RerankConfig,
    pub deepseek: DeepSeekConfig,
    pub server: ServerConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            inputs: PathBuf::from("dicts"),
            data_dir: PathBuf::from("data"),
            store_path: PathBuf::from("data/index.sqlite"),
            models_dir: PathBuf::from("models"),
            embedding: EmbeddingConfig::default(),
            chunking: ChunkingConfig::default(),
            cleaning: CleaningConfig::default(),
            rerank: RerankConfig::default(),
            deepseek: DeepSeekConfig::default(),
            server: ServerConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub model: String,
    pub max_length: Option<usize>,
    /// кол-во потоков CPU для ONNX-вычислений
    pub intra_threads: usize,
    /// Размер батча при индексации. Для крупных моделей (например, bge-m3,
    /// ~2.3 ГБ) большие батчи требуют много памяти, поэтому значение вынесено
    /// в конфигурацию.
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: String::from("bge-small-en-v1.5"),
            max_length: Some(512),
            intra_threads: 2,
            batch_size: 128,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ChunkingConfig {
    pub target_tokens: usize,
    pub overlap_tokens: usize,
    pub max_chunk_chars: usize,
    /// Чанки короче этого размера (в символах) склеиваются с соседними
    /// в пределах одного раздела — иначе короткие подразделы дают
    /// неинформативные фрагменты по 200-500 символов.
    pub min_chunk_chars: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            target_tokens: 900,
            overlap_tokens: 150,
            max_chunk_chars: 4000,
            min_chunk_chars: 600,
        }
    }
}

/// Настройки очистки PDF-текста (см. `crate::clean`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct CleaningConfig {
    /// выключить очистку целиком (для абляций «сырой текст» vs «очищенный»)
    pub enabled: bool,
    /// типографские символы -> ASCII, удаление лигатур/nbsp/soft hyphen
    pub normalize_unicode: bool,
    /// склейка слов, разорванных переносом
    pub dehyphenate: bool,
    /// удаление строк оглавления (точечные выноски)
    pub strip_toc: bool,
    /// удаление колонтитулов и повторяющихся строк
    pub remove_boilerplate: bool,
    /// переразметка схлопнутых таблиц: `a  b  c` -> `a | b | c`
    pub render_table_rows: bool,
    /// сколько разрывов из 2+ пробелов считать признаком табличной строки
    pub table_gap_threshold: usize,
    /// строка считается колонтитулом, если встречается минимум на стольких страницах
    pub boilerplate_min_pages: usize,
    /// ... или на такой доле страниц документа
    pub boilerplate_page_ratio: f64,
}

impl Default for CleaningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            normalize_unicode: true,
            dehyphenate: true,
            strip_toc: true,
            remove_boilerplate: true,
            render_table_rows: true,
            table_gap_threshold: 2,
            boilerplate_min_pages: 5,
            boilerplate_page_ratio: 0.15,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RerankConfig {
    pub enabled: bool,
    pub model: String,
    pub candidates: usize,
    pub final_top_k: usize,
    pub intra_threads: usize,
    /// Максимальная длина пары «запрос + фрагмент» для кросс-энкодера.
    /// Стоимость реранка линейна по числу кандидатов и по этой длине:
    /// 50 кандидатов по 512 токенов на слабом CPU занимают ~2 минуты на запрос.
    pub max_length: usize,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: String::from("bge-reranker-base"),
            candidates: 50,
            final_top_k: 6,
            intra_threads: 2,
            max_length: 512,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DeepSeekConfig {
    pub api_key: Option<String>,
    pub api_key_env: String,
    pub base_url: String,
    pub chat_path: String,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub timeout_secs: u64,
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            api_key_env: String::from("DEEPSEEK_API_KEY"),
            base_url: String::from("https://api.deepseek.com"),
            chat_path: String::from("/chat/completions"),
            model: String::from("deepseek-chat"),
            temperature: 0.2,
            max_tokens: 1500,
            timeout_secs: 120,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub bind: String,
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: String::from("127.0.0.1"),
            port: 8080,
        }
    }
}

impl Config {
    /// Загружает конфиг: default -> config.toml (если есть) -> явный путь -> env-переменные.
    pub fn load(path: Option<&str>) -> Result<Config> {
        let mut cfg = match path {
            Some(p) => {
                let text = std::fs::read_to_string(p)
                    .with_context(|| format!("cannot read config file: {p}"))?;
                toml::from_str::<Config>(&text)
                    .with_context(|| format!("invalid config file: {p}"))?
            }
            None => {
                if PathBuf::from("config.toml").exists() {
                    let text = std::fs::read_to_string("config.toml")?;
                    toml::from_str::<Config>(&text).context("invalid config.toml")?
                } else {
                    Config::default()
                }
            }
        };

        if cfg.deepseek.api_key.is_none() {
            if let Ok(k) = env::var(&cfg.deepseek.api_key_env) {
                cfg.deepseek.api_key = Some(k);
            }
        }

        Ok(cfg)
    }

    pub fn require_api_key(&self) -> Result<String> {
        self.deepseek
            .api_key
            .clone()
            .context("DeepSeek API key not found: set the DEEPSEEK_API_KEY environment variable or api_key in config.toml")
    }
}