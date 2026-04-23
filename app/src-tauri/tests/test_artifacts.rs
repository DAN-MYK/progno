#[cfg(test)]
mod tests {
    use std::fs;
    use serde_json::json;
    use progno::artifacts::{get_data_as_of, get_player_elo, get_player_surface_matches, load_elo_state};

    #[test]
    fn test_load_elo_state_parses_json() {
        // Create a temp JSON file
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
        let state = json!({
            "data_as_of": "2026-04-20",
            "players": {}
        });

        let date = get_data_as_of(&state);
        assert_eq!(date, "2026-04-20");
    }

    #[test]
    fn test_get_data_as_of_missing() {
        let state = json!({
            "players": {}
        });

        let date = get_data_as_of(&state);
        assert_eq!(date, "unknown");
    }
}
