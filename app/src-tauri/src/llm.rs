use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::parser::ParsedMatch;

const GEMINI_URL: &str =
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent";
const GROK_URL: &str = "https://api.x.ai/v1/chat/completions";

fn build_prompt(text: &str) -> String {
    format!(
        r#"Extract tennis matches from the text below.
Return a JSON object with a "matches" array. Each element must have:
  "player_a"  – last name of the first player (string)
  "player_b"  – last name of the second player (string)
  "surface"   – one of "Hard", "Clay", "Grass" (default "Hard" when unknown)

Text:
{}

Return ONLY the JSON object, no markdown, no explanation.
Example: {{"matches":[{{"player_a":"Alcaraz","player_b":"Sinner","surface":"Clay"}}]}}"#,
        text
    )
}

#[derive(Deserialize)]
struct LlmMatch {
    player_a: String,
    player_b: String,
    surface: String,
}

#[derive(Deserialize)]
struct LlmResponse {
    matches: Vec<LlmMatch>,
}

fn parse_llm_json(raw: &str) -> Result<Vec<ParsedMatch>> {
    // strip optional markdown fences
    let cleaned = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let resp: LlmResponse = serde_json::from_str(cleaned)
        .map_err(|e| anyhow!("LLM JSON parse error: {e} — raw: {cleaned}"))?;

    Ok(resp
        .matches
        .into_iter()
        .map(|m| ParsedMatch {
            player_a: m.player_a,
            player_b: m.player_b,
            surface: normalize_surface(&m.surface),
        })
        .collect())
}

fn normalize_surface(s: &str) -> String {
    match s.to_lowercase().as_str() {
        "clay" => "Clay".to_string(),
        "grass" => "Grass".to_string(),
        _ => "Hard".to_string(),
    }
}

async fn call_gemini(text: &str, api_key: &str) -> Result<Vec<ParsedMatch>> {
    let url = format!("{}?key={}", GEMINI_URL, api_key);
    let body = serde_json::json!({
        "contents": [{"parts": [{"text": build_prompt(text)}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    });

    let resp = Client::new()
        .post(&url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    let content = resp["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| anyhow!("Gemini: unexpected response shape: {resp}"))?;

    parse_llm_json(content)
}

async fn call_grok(text: &str, api_key: &str) -> Result<Vec<ParsedMatch>> {
    let body = serde_json::json!({
        "model": "grok-3-mini",
        "messages": [{"role": "user", "content": build_prompt(text)}],
        "response_format": {"type": "json_object"}
    });

    let resp = Client::new()
        .post(GROK_URL)
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow!("Grok: unexpected response shape: {resp}"))?;

    parse_llm_json(content)
}

pub async fn parse_with_llm(text: &str, provider: &str, api_key: &str) -> Result<Vec<ParsedMatch>> {
    match provider {
        "grok" => call_grok(text, api_key).await,
        _ => call_gemini(text, api_key).await,
    }
}

// ── News / injury check ──────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NewsCheckResult {
    pub injury_flag: bool,
    pub summary: String,
    pub items: Vec<String>,
}

fn build_news_prompt(player_a: &str, player_b: &str) -> String {
    format!(
        r#"You are a tennis analyst. Search your knowledge for recent news (last 30 days) about these two players: {player_a} and {player_b}.

Focus on: injuries, illness, withdrawals, fitness issues, recent form (last 3 matches), and any other pre-match concerns.

Return a JSON object with:
  "injury_flag": true if either player has a known or suspected injury/fitness concern, false otherwise
  "summary": one sentence summarising the key finding (or "No significant news found." if nothing notable)
  "items": array of bullet-point strings with specific findings (max 5), each prefixed with the player name

Return ONLY the JSON object, no markdown."#
    )
}

async fn news_gemini(player_a: &str, player_b: &str, api_key: &str) -> Result<NewsCheckResult> {
    let url = format!("{}?key={}", GEMINI_URL, api_key);
    let body = serde_json::json!({
        "contents": [{"parts": [{"text": build_news_prompt(player_a, player_b)}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    });

    let resp = Client::new()
        .post(&url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    let content = resp["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| anyhow!("Gemini news: unexpected response shape"))?;

    let cleaned = content.trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    serde_json::from_str(cleaned).map_err(|e| anyhow!("News JSON parse error: {e} — raw: {cleaned}"))
}

async fn news_grok(player_a: &str, player_b: &str, api_key: &str) -> Result<NewsCheckResult> {
    let body = serde_json::json!({
        "model": "grok-3",
        "messages": [{"role": "user", "content": build_news_prompt(player_a, player_b)}],
        "response_format": {"type": "json_object"}
    });

    let resp = Client::new()
        .post(GROK_URL)
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow!("Grok news: unexpected response shape"))?;

    serde_json::from_str(content).map_err(|e| anyhow!("News JSON parse error: {e}"))
}

pub async fn check_news(
    player_a: &str,
    player_b: &str,
    provider: &str,
    api_key: &str,
) -> Result<NewsCheckResult> {
    match provider {
        "grok" => news_grok(player_a, player_b, api_key).await,
        _ => news_gemini(player_a, player_b, api_key).await,
    }
}

#[tauri::command]
pub async fn check_player_news(
    player_a: String,
    player_b: String,
    provider: String,
    api_key: String,
) -> Result<NewsCheckResult, String> {
    check_news(&player_a, &player_b, &provider, &api_key)
        .await
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_llm_json_basic() {
        let raw = r#"{"matches":[{"player_a":"Alcaraz","player_b":"Sinner","surface":"Clay"}]}"#;
        let result = parse_llm_json(raw).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].player_a, "Alcaraz");
        assert_eq!(result[0].surface, "Clay");
    }

    #[test]
    fn test_parse_llm_json_strips_fences() {
        let raw = "```json\n{\"matches\":[{\"player_a\":\"Djokovic\",\"player_b\":\"Nadal\",\"surface\":\"Grass\"}]}\n```";
        let result = parse_llm_json(raw).unwrap();
        assert_eq!(result[0].surface, "Grass");
    }

    #[test]
    fn test_normalize_surface_unknown() {
        assert_eq!(normalize_surface("indoor"), "Hard");
        assert_eq!(normalize_surface("CLAY"), "Clay");
        assert_eq!(normalize_surface("Grass"), "Grass");
    }

    #[test]
    fn test_parse_llm_json_multiple() {
        let raw = r#"{"matches":[
            {"player_a":"Zverev","player_b":"Ruud","surface":"Clay"},
            {"player_a":"Fritz","player_b":"Tiafoe","surface":"Hard"}
        ]}"#;
        let result = parse_llm_json(raw).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[1].player_a, "Fritz");
    }

    #[test]
    fn test_parse_llm_json_empty_matches() {
        let raw = r#"{"matches":[]}"#;
        let result = parse_llm_json(raw).unwrap();
        assert_eq!(result.len(), 0);
    }
}
