#[cfg(test)]
mod tests {
    use serde_json::json;
    use progno::commands::{parse_and_predict, PredictResponse};

    #[test]
    fn test_parse_and_predict_returns_json() {
        let text = "Alcaraz vs Sinner - Clay";
        let elo_json = json!({
            "data_as_of": "2026-04-20",
            "players": {
                "alcaraz_carlos": {
                    "elo_overall": 1600,
                    "elo_hard": 1650,
                    "elo_clay": 1550,
                    "elo_grass": 1500,
                    "matches_played": 10,
                    "matches_played_hard": 5,
                    "matches_played_clay": 3,
                    "matches_played_grass": 2
                },
                "sinner_jannik": {
                    "elo_overall": 1500,
                    "elo_hard": 1500,
                    "elo_clay": 1450,
                    "elo_grass": 1500,
                    "matches_played": 8,
                    "matches_played_hard": 4,
                    "matches_played_clay": 2,
                    "matches_played_grass": 2
                }
            }
        }).to_string();

        let response = parse_and_predict(text.to_string(), elo_json);
        assert_eq!(response.predictions.len(), 1);
        assert_eq!(response.data_as_of, "2026-04-20");
        assert!(response.error.is_none());
        assert!(response.predictions[0].prob_a_wins > 0.5);
    }

    #[test]
    fn test_parse_and_predict_invalid_json() {
        let text = "Alcaraz vs Sinner";
        let elo_json = "not valid json";

        let response = parse_and_predict(text.to_string(), elo_json.to_string());
        assert_eq!(response.predictions.len(), 0);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_parse_and_predict_missing_player() {
        let text = "Unknown vs Sinner";
        let elo_json = json!({
            "data_as_of": "2026-04-20",
            "players": {}
        }).to_string();

        let response = parse_and_predict(text.to_string(), elo_json);
        assert_eq!(response.predictions.len(), 0);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_parse_and_predict_multiple() {
        let text = "Alcaraz vs Sinner\nDjokovic vs Medvedev";
        let elo_json = json!({
            "data_as_of": "2026-04-20",
            "players": {
                "alcaraz_carlos": { "elo_overall": 1600, "elo_hard": 1600, "elo_clay": 1600, "elo_grass": 1600, "matches_played": 10, "matches_played_hard": 5, "matches_played_clay": 5, "matches_played_grass": 0 },
                "sinner_jannik": { "elo_overall": 1500, "elo_hard": 1500, "elo_clay": 1500, "elo_grass": 1500, "matches_played": 8, "matches_played_hard": 4, "matches_played_clay": 4, "matches_played_grass": 0 },
                "djokovic_novak": { "elo_overall": 1700, "elo_hard": 1700, "elo_clay": 1700, "elo_grass": 1700, "matches_played": 20, "matches_played_hard": 10, "matches_played_clay": 5, "matches_played_grass": 5 },
                "medvedev_daniil": { "elo_overall": 1400, "elo_hard": 1400, "elo_clay": 1400, "elo_grass": 1400, "matches_played": 8, "matches_played_hard": 4, "matches_played_clay": 2, "matches_played_grass": 2 }
            }
        }).to_string();

        let response = parse_and_predict(text.to_string(), elo_json);
        assert_eq!(response.predictions.len(), 2);
        assert_eq!(response.predictions[0].player_a, "Alcaraz");
        assert_eq!(response.predictions[1].player_a, "Djokovic");
    }

    #[test]
    fn test_parse_and_predict_empty_input() {
        let text = "";
        let elo_json = json!({
            "data_as_of": "2026-04-20",
            "players": {}
        }).to_string();

        let response = parse_and_predict(text.to_string(), elo_json);
        assert_eq!(response.predictions.len(), 0);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_parse_and_predict_probability_bounds() {
        let text = "Alcaraz vs Sinner";
        let elo_json = json!({
            "data_as_of": "2026-04-20",
            "players": {
                "alcaraz_carlos": { "elo_overall": 1600, "elo_hard": 1600, "elo_clay": 1600, "elo_grass": 1600, "matches_played": 10, "matches_played_hard": 5, "matches_played_clay": 5, "matches_played_grass": 0 },
                "sinner_jannik": { "elo_overall": 1500, "elo_hard": 1500, "elo_clay": 1500, "elo_grass": 1500, "matches_played": 8, "matches_played_hard": 4, "matches_played_clay": 4, "matches_played_grass": 0 }
            }
        }).to_string();

        let response = parse_and_predict(text.to_string(), elo_json);
        assert_eq!(response.predictions.len(), 1);
        let pred = &response.predictions[0];

        // Probability bounds check
        assert!(pred.prob_a_wins >= 0.0 && pred.prob_a_wins <= 1.0);
        assert!(pred.prob_b_wins >= 0.0 && pred.prob_b_wins <= 1.0);
        // Sum should be 1.0
        assert!((pred.prob_a_wins + pred.prob_b_wins - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_get_data_as_of_command() {
        let elo_json = json!({
            "data_as_of": "2026-04-20",
            "players": {}
        }).to_string();

        let date = crate::commands::get_data_as_of_cmd(elo_json);
        assert_eq!(date, "2026-04-20");
    }
}
