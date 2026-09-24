use crate::config::DeepSeekConfig;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: RespMessage,
}

#[derive(Deserialize)]
struct RespMessage {
    content: serde_json::Value,
}

pub struct DeepSeekClient {
    http: reqwest::Client,
    base_url: String,
    chat_path: String,
    api_key: Option<String>,
    model: String,
    temperature: f32,
    max_tokens: u32,
}

impl DeepSeekClient {
    pub fn new(cfg: &DeepSeekConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs.max(1)))
            .build()
            .context("failed to build http client")?;
        Ok(Self {
            http,
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            chat_path: cfg.chat_path.clone(),
            api_key: cfg.api_key.clone(),
            model: cfg.model.clone(),
            temperature: cfg.temperature,
            max_tokens: cfg.max_tokens,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn chat(&self, system: &str, user: &str) -> Result<String> {
        self.chat_impl(system, user, None).await
    }

    /// Режим строгого JSON (`response_format: json_object`) — для LLM-судьи.
    pub async fn chat_json(&self, system: &str, user: &str) -> Result<String> {
        self.chat_impl(
            system,
            user,
            Some(serde_json::json!({ "type": "json_object" })),
        )
        .await
    }

    async fn chat_impl(
        &self,
        system: &str,
        user: &str,
        response_format: Option<serde_json::Value>,
    ) -> Result<String> {
        let api_key = self
            .api_key
            .as_deref()
            .context("DeepSeek API key not found: set the DEEPSEEK_API_KEY environment variable or api_key in config.toml")?;
        let url = format!("{}{}", self.base_url, self.chat_path);
        let payload = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user.to_string(),
                },
            ],
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream: false,
            response_format,
        };
        let resp = self
            .http
            .post(&url)
            .bearer_auth(api_key)
            .json(&payload)
            .send()
            .await
            .context("DeepSeek request failed")?;
        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            bail!("DeepSeek API error {status}: {body}");
        }
        let parsed: ChatResponse = serde_json::from_str(&body).context("invalid DeepSeek response")?;
        match parsed.choices.into_iter().next() {
            Some(c) => match c.message.content {
                serde_json::Value::String(s) => Ok(s),
                other => Ok(other.to_string()),
            },
            None => bail!("DeepSeek returned no choices"),
        }
    }
}