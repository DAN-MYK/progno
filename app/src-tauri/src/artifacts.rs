use serde_json::Value;
use std::fs;

/// Load Elo state from JSON file.
pub fn load_elo_state(path: &str) -> Result<Value, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path, e))?;

    let value: Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;

    Ok(value)
}

/// Get Elo rating for a player on a specific surface.
pub fn get_player_elo(state: &Value, player_id: &str, surface: &str) -> Result<f64, String> {
    state
        .get("players")
        .and_then(|p| p.get(player_id))
        .and_then(|player| {
            let field = match surface.to_lowercase().as_str() {
                "hard" => "elo_hard",
                "clay" => "elo_clay",
                "grass" => "elo_grass",
                _ => "elo_overall",
            };
            player.get(field).and_then(|v| v.as_f64())
        })
        .ok_or_else(|| format!("Player {} has no {} Elo", player_id, surface))
}

/// Get number of matches played on a specific surface.
pub fn get_player_surface_matches(state: &Value, player_id: &str, surface: &str) -> Result<u32, String> {
    state
        .get("players")
        .and_then(|p| p.get(player_id))
        .and_then(|player| {
            let field = match surface.to_lowercase().as_str() {
                "hard" => "matches_played_hard",
                "clay" => "matches_played_clay",
                "grass" => "matches_played_grass",
                _ => "matches_played",
            };
            player.get(field).and_then(|v| v.as_u64())
        })
        .ok_or_else(|| format!("Player {} has no surface match count", player_id))
        .map(|v| v as u32)
}

/// Get the date the model data is as of.
pub fn get_data_as_of(state: &Value) -> String {
    state
        .get("data_as_of")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string()
}
