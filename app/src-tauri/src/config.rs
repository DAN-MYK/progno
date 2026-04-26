use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Serialize, Deserialize, Default, Debug)]
pub struct ApiKeys {
    #[serde(default)]
    pub gemini: String,
    #[serde(default)]
    pub grok: String,
    #[serde(default)]
    pub rapidapi: String,
    #[serde(default)]
    pub api_tennis: String,
}

/// Load API keys from ~/.progno_keys.json (outside the repo, never committed).
/// Returns empty strings for missing keys — app falls back to whatever is in localStorage.
#[tauri::command]
pub fn load_api_keys() -> Result<ApiKeys, String> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let path = std::path::Path::new(&home).join(".progno_keys.json");

    if !path.exists() {
        return Ok(ApiKeys::default());
    }

    let content = fs::read_to_string(&path)
        .map_err(|e| format!("Cannot read ~/.progno_keys.json: {e}"))?;

    serde_json::from_str(&content)
        .map_err(|e| format!("Cannot parse ~/.progno_keys.json: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_keys_default() {
        let k = ApiKeys::default();
        assert!(k.gemini.is_empty());
        assert!(k.grok.is_empty());
    }

    #[test]
    fn test_api_keys_partial_json() {
        let json = r#"{"gemini":"abc"}"#;
        let k: ApiKeys = serde_json::from_str(json).unwrap();
        assert_eq!(k.gemini, "abc");
        assert!(k.grok.is_empty());
    }
}
