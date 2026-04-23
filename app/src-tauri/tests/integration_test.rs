// Integration tests for the complete Kelly flow

use progno::commands::*;
use serde_json::json;

// Scenario 1: Basic Kelly calculation
#[test]
fn test_scenario_1_basic_kelly() {
    let state = json!({
        "data_as_of": "2026-04-20",
        "players": {
            "alcaraz": {
                "elo_overall": 1600,
                "elo_hard": 1650,
                "elo_clay": 1550,
                "elo_grass": 1500,
                "matches_played": 10,
                "matches_played_hard": 5,
                "matches_played_clay": 3,
                "matches_played_grass": 2
            },
            "sinner": {
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
    });

    let text = "Alcaraz vs Sinner - Hard";
    let response = predict_text(text, &state);

    assert_eq!(response.predictions.len(), 1);
    assert!(response.error.is_none());

    let pred = &response.predictions[0];
    assert_eq!(pred.player_a, "Alcaraz");
    assert_eq!(pred.player_b, "Sinner");
    assert_eq!(pred.surface, "Hard");
    assert!(pred.prob_a_wins > 0.5);
    assert!(pred.elo_a_overall > 0.0);
    assert!(pred.elo_b_overall > 0.0);

    // Kelly calculation with odds 2.50
    let req = KellyRequest {
        model_prob: pred.prob_a_wins,
        decimal_odds: 2.50,
        bankroll: 1000.0,
        kelly_fraction: 0.25,
    };

    let result = calculate_kelly_impl(req).unwrap();
    assert!((result.implied_prob - 0.4).abs() < 0.001);
    assert!(result.edge > 0.0);
    assert!(result.stake > 0.0);
}

// Scenario 2: Negative edge (bad bet)
#[test]
fn test_scenario_2_negative_edge() {
    let req = KellyRequest {
        model_prob: 0.6,
        decimal_odds: 1.20,
        bankroll: 1000.0,
        kelly_fraction: 0.25,
    };

    let result = calculate_kelly_impl(req).unwrap();
    assert!((result.implied_prob - 0.8333).abs() < 0.01);
    assert!(result.edge < 0.0);
    assert_eq!(result.stake, 0.0);
}

// Scenario 3: Bankroll scaling
#[test]
fn test_scenario_3_bankroll_scaling() {
    let req_1000 = KellyRequest {
        model_prob: 0.6,
        decimal_odds: 2.50,
        bankroll: 1000.0,
        kelly_fraction: 0.25,
    };

    let result_1000 = calculate_kelly_impl(req_1000).unwrap();
    let stake_1000 = result_1000.stake;

    let req_2000 = KellyRequest {
        model_prob: 0.6,
        decimal_odds: 2.50,
        bankroll: 2000.0,
        kelly_fraction: 0.25,
    };

    let result_2000 = calculate_kelly_impl(req_2000).unwrap();
    let stake_2000 = result_2000.stake;

    assert!((stake_2000 - 2.0 * stake_1000).abs() < 0.01);
}

// Scenario 4: Kelly fraction scaling
#[test]
fn test_scenario_4_kelly_fraction_scaling() {
    let req_01x = KellyRequest {
        model_prob: 0.6,
        decimal_odds: 2.50,
        bankroll: 1000.0,
        kelly_fraction: 0.1,
    };

    let result_01x = calculate_kelly_impl(req_01x).unwrap();

    let req_025x = KellyRequest {
        model_prob: 0.6,
        decimal_odds: 2.50,
        bankroll: 1000.0,
        kelly_fraction: 0.25,
    };

    let result_025x = calculate_kelly_impl(req_025x).unwrap();

    let req_05x = KellyRequest {
        model_prob: 0.6,
        decimal_odds: 2.50,
        bankroll: 1000.0,
        kelly_fraction: 0.5,
    };

    let result_05x = calculate_kelly_impl(req_05x).unwrap();

    let req_1x = KellyRequest {
        model_prob: 0.6,
        decimal_odds: 2.50,
        bankroll: 1000.0,
        kelly_fraction: 1.0,
    };

    let result_1x = calculate_kelly_impl(req_1x).unwrap();

    assert!((result_025x.stake - 2.5 * result_01x.stake).abs() < 0.01);
    assert!((result_05x.stake - 2.0 * result_025x.stake).abs() < 0.01);
    assert!((result_1x.stake - 2.0 * result_05x.stake).abs() < 0.01);
}

// Scenario 5: Edge display
#[test]
fn test_scenario_5_edge_display() {
    let req_positive = KellyRequest {
        model_prob: 0.6,
        decimal_odds: 2.50,
        bankroll: 1000.0,
        kelly_fraction: 0.25,
    };

    let result_pos = calculate_kelly_impl(req_positive).unwrap();
    assert!(result_pos.edge > 0.0);

    let req_negative = KellyRequest {
        model_prob: 0.3,
        decimal_odds: 2.50,
        bankroll: 1000.0,
        kelly_fraction: 0.25,
    };

    let result_neg = calculate_kelly_impl(req_negative).unwrap();
    assert!(result_neg.edge < 0.0);

    let req_zero = KellyRequest {
        model_prob: 0.4,
        decimal_odds: 2.50,
        bankroll: 1000.0,
        kelly_fraction: 0.25,
    };

    let result_zero = calculate_kelly_impl(req_zero).unwrap();
    assert!(result_zero.edge.abs() < 0.001);
}

// Scenario 6: Multiple matches
#[test]
fn test_scenario_6_multiple_matches() {
    let state = json!({
        "data_as_of": "2026-04-20",
        "players": {
            "alcaraz": {
                "elo_overall": 1600, "elo_hard": 1650, "elo_clay": 1550, "elo_grass": 1500,
                "matches_played": 10, "matches_played_hard": 5, "matches_played_clay": 3, "matches_played_grass": 2
            },
            "sinner": {
                "elo_overall": 1500, "elo_hard": 1500, "elo_clay": 1450, "elo_grass": 1500,
                "matches_played": 8, "matches_played_hard": 4, "matches_played_clay": 2, "matches_played_grass": 2
            },
            "djokovic": {
                "elo_overall": 1700, "elo_hard": 1700, "elo_clay": 1700, "elo_grass": 1700,
                "matches_played": 20, "matches_played_hard": 10, "matches_played_clay": 5, "matches_played_grass": 5
            },
            "medvedev": {
                "elo_overall": 1400, "elo_hard": 1400, "elo_clay": 1400, "elo_grass": 1400,
                "matches_played": 8, "matches_played_hard": 4, "matches_played_clay": 2, "matches_played_grass": 2
            }
        }
    });

    let text = "Alcaraz vs Sinner - Hard\nDjokovic vs Medvedev - Hard";
    let response = predict_text(text, &state);

    assert_eq!(response.predictions.len(), 2);
    assert!(response.error.is_none());

    let req1 = KellyRequest {
        model_prob: response.predictions[0].prob_a_wins,
        decimal_odds: 2.50,
        bankroll: 1000.0,
        kelly_fraction: 0.25,
    };

    let req2 = KellyRequest {
        model_prob: response.predictions[1].prob_a_wins,
        decimal_odds: 3.00,
        bankroll: 1000.0,
        kelly_fraction: 0.25,
    };

    let result1 = calculate_kelly_impl(req1).unwrap();
    let result2 = calculate_kelly_impl(req2).unwrap();

    assert!(result1.stake >= 0.0);
    assert!(result2.stake >= 0.0);
}

// Scenario 7: Empty input error handling
#[test]
fn test_scenario_7_empty_input() {
    let state = json!({"data_as_of": "2026-04-20", "players": {}});
    let response = predict_text("", &state);

    assert_eq!(response.predictions.len(), 0);
    assert!(response.error.is_some());
}
