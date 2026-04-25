use serde_json::Value;
use std::fs;
use std::path::PathBuf;

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

/// Load elo_state.json for a specific tour from artifacts/{tour}/elo_state.json
/// relative to the binary location.
pub fn load_elo_state_for_tour(tour: &str) -> Result<serde_json::Value, String> {
    let path = PathBuf::from(format!("artifacts/{}/elo_state.json", tour));
    load_elo_state(path.to_str().unwrap_or("elo_state.json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use serde_json::json;

    #[test]
    fn test_load_elo_state_parses_json() {
        let tmp_dir = std::env::temp_dir();
        let json_path = tmp_dir.join("test_elo_state.json");
        let content = r#"{"data_as_of": "2026-04-20", "players": {"1": {"elo_overall": 1600, "elo_hard": 1650, "elo_clay": 1500, "elo_grass": 1500, "matches_played": 10, "matches_played_hard": 5, "matches_played_clay": 3, "matches_played_grass": 2}}}"#;
        fs::write(&json_path, content).unwrap();

        let state = load_elo_state(json_path.to_str().unwrap()).unwrap();
        assert!(state.get("data_as_of").is_some());
        assert!(state.get("players").is_some());

        fs::remove_file(&json_path).ok();
    }

    #[test]
    fn test_load_elo_state_missing_file() {
        let result = load_elo_state("/nonexistent/path/elo_state.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_elo_state_invalid_json() {
        let tmp_dir = std::env::temp_dir();
        let json_path = tmp_dir.join("test_invalid.json");
        fs::write(&json_path, "not valid json {").unwrap();

        let result = load_elo_state(json_path.to_str().unwrap());
        assert!(result.is_err());

        fs::remove_file(&json_path).ok();
    }

    #[test]
    fn test_get_player_elo_overall() {
        let state = json!({
            "players": {
                "1": {
                    "elo_overall": 1600,
                    "elo_hard": 1650,
                    "elo_clay": 1500,
                    "elo_grass": 1500,
                    "matches_played": 10
                }
            }
        });
        let elo = get_player_elo(&state, "1", "overall").unwrap();
        assert_eq!(elo, 1600.0);
    }

    #[test]
    fn test_get_player_elo_surface() {
        let state = json!({
            "players": {
                "1": {
                    "elo_overall": 1600,
                    "elo_hard": 1650,
                    "elo_clay": 1500,
                    "elo_grass": 1500
                }
            }
        });
        let elo_hard = get_player_elo(&state, "1", "hard").unwrap();
        assert_eq!(elo_hard, 1650.0);
        let elo_clay = get_player_elo(&state, "1", "clay").unwrap();
        assert_eq!(elo_clay, 1500.0);
    }

    #[test]
    fn test_get_player_surface_matches() {
        let state = json!({
            "players": {
                "1": {
                    "matches_played": 10,
                    "matches_played_hard": 5,
                    "matches_played_clay": 3,
                    "matches_played_grass": 2
                }
            }
        });
        let matches = get_player_surface_matches(&state, "1", "hard").unwrap();
        assert_eq!(matches, 5);
    }

    #[test]
    fn test_get_data_as_of() {
        let state = json!({"data_as_of": "2026-04-20", "players": {}});
        let date = get_data_as_of(&state);
        assert_eq!(date, "2026-04-20");
    }

    #[test]
    fn test_get_data_as_of_missing() {
        let state = json!({"players": {}});
        let date = get_data_as_of(&state);
        assert_eq!(date, "unknown");
    }
}
