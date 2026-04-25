use serde::{Deserialize, Serialize};
use crate::artifacts::{get_data_as_of, get_player_elo, get_player_surface_matches};
use crate::elo::{expected_probability, surface_elo};
use crate::parser::{parse_match_text, ParsedMatch};
use crate::kelly;
#[cfg(not(test))]
use crate::state::AppState;

#[derive(Serialize, Deserialize, Clone)]
pub struct KellyRequest {
    pub model_prob: f64,
    pub decimal_odds: f64,
    pub bankroll: f64,
    pub kelly_fraction: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct KellyResult {
    pub implied_prob: f64,
    pub edge: f64,
    pub full_kelly: f64,
    pub fractional_kelly: f64,
    pub stake: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PredictionResult {
    pub player_a: String,
    pub player_b: String,
    pub surface: String,
    pub prob_a_wins: f64,
    pub prob_b_wins: f64,
    pub elo_a_overall: f64,
    pub elo_b_overall: f64,
    pub ml_prob_a_wins: Option<f64>,
    pub confidence_flag: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct PredictResponse {
    pub predictions: Vec<PredictionResult>,
    pub data_as_of: String,
    pub error: Option<String>,
}

pub fn predict_text(text: &str, elo_state: &serde_json::Value) -> PredictResponse {
    let matches = match parse_match_text(text) {
        Ok(m) => m,
        Err(e) => return PredictResponse {
            predictions: vec![],
            data_as_of: "unknown".to_string(),
            error: Some(e),
        },
    };

    let data_as_of = get_data_as_of(elo_state);
    let mut predictions = Vec::new();

    for m in matches {
        match predict_match(&m, elo_state) {
            Ok(pred) => predictions.push(pred),
            Err(e) => eprintln!("Failed to predict {} vs {}: {}", m.player_a, m.player_b, e),
        }
    }

    let error = if predictions.is_empty() {
        Some("No matches could be predicted".to_string())
    } else {
        None
    };
    PredictResponse { predictions, data_as_of, error }
}

#[cfg(not(test))]
#[tauri::command]
pub fn parse_and_predict(
    text: String,
    tour: String,
    app_state: tauri::State<AppState>,
) -> PredictResponse {
    let elo = match tour.as_str() {
        "wta" => app_state.elo_wta.lock().unwrap(),
        _ => app_state.elo_atp.lock().unwrap(),
    };
    match &*elo {
        None => PredictResponse {
            predictions: vec![],
            data_as_of: "unknown".to_string(),
            error: Some(format!("Elo data for {} not loaded. Run 'just elo' first.", tour.to_uppercase())),
        },
        Some(state) => predict_text(&text, state),
    }
}

#[cfg(not(test))]
#[tauri::command]
pub fn get_data_as_of_cmd(tour: String, app_state: tauri::State<AppState>) -> String {
    let elo = match tour.as_str() {
        "wta" => app_state.elo_wta.lock().unwrap(),
        _ => app_state.elo_atp.lock().unwrap(),
    };
    match &*elo {
        None => "unknown".to_string(),
        Some(state) => get_data_as_of(state),
    }
}

#[cfg(not(test))]
#[tauri::command]
pub fn calculate_kelly(request: KellyRequest) -> Result<KellyResult, String> {
    calculate_kelly_impl(request)
}

#[derive(Serialize, Deserialize)]
pub struct MlPredictRequest {
    pub text: String,
    pub tour: String,
    pub tourney_date: String,
}

#[cfg(not(test))]
#[tauri::command]
pub async fn predict_with_ml(
    request: MlPredictRequest,
    app_state: tauri::State<'_, AppState>,
    sidecar_state: tauri::State<'_, std::sync::Mutex<crate::sidecar::SidecarState>>,
) -> Result<PredictResponse, String> {
    use crate::sidecar::{MlMatchRequest, ml_predict};

    // Compute elo_resp and drop the MutexGuard before any await points
    let elo_resp = {
        let elo_guard = match request.tour.as_str() {
            "wta" => app_state.elo_wta.lock().unwrap(),
            _ => app_state.elo_atp.lock().unwrap(),
        };
        match &*elo_guard {
            None => return Err(format!("Elo data for {} not loaded", request.tour.to_uppercase())),
            Some(elo) => predict_text(&request.text, elo),
        }
    }; // elo_guard dropped here

    let port = sidecar_state.lock().unwrap().port;
    if port.is_none() {
        return Ok(elo_resp);
    }
    let port = port.unwrap();

    let ml_matches: Vec<MlMatchRequest> = elo_resp.predictions.iter().map(|p| MlMatchRequest {
        tour: request.tour.clone(),
        player_a_id: normalize_player_id(&p.player_a),
        player_b_id: normalize_player_id(&p.player_b),
        surface: p.surface.clone(),
        // Note: parser.rs only extracts player names and surface from pasted text.
        // Tournament metadata (level, round, best_of) is unavailable from paste input.
        // These defaults (ATP 250, R32, BO3) are used for all matches.
        // TODO(Phase 5): allow users to specify tournament context for better model calibration.
        tourney_level: "A".to_string(),
        round_: "R32".to_string(),
        best_of: 3,
        tourney_date: request.tourney_date.clone(),
    }).collect();

    match ml_predict(port, ml_matches).await {
        Ok(ml_resp) => {
            let enriched: Vec<PredictionResult> = elo_resp.predictions.into_iter()
                .zip(ml_resp.predictions.into_iter())
                .map(|(mut elo_pred, ml_pred)| {
                    elo_pred.ml_prob_a_wins = Some(ml_pred.prob_a_wins);
                    elo_pred.confidence_flag = Some(ml_pred.confidence_flag);
                    elo_pred
                })
                .collect();
            Ok(PredictResponse { predictions: enriched, ..elo_resp })
        }
        Err(e) => {
            eprintln!("[ml] predict failed: {e}, falling back to Elo");
            Ok(elo_resp)
        }
    }
}

#[cfg(not(test))]
#[tauri::command]
pub async fn parse_with_llm(
    text: String,
    tour: String,
    provider: String,
    api_key: String,
    app_state: tauri::State<'_, AppState>,
) -> Result<PredictResponse, String> {
    let matches = crate::llm::parse_with_llm(&text, &provider, &api_key)
        .await
        .map_err(|e| e.to_string())?;

    if matches.is_empty() {
        return Ok(PredictResponse {
            predictions: vec![],
            data_as_of: "unknown".to_string(),
            error: Some("AI could not extract any matches from this text.".to_string()),
        });
    }

    // Build synthetic text from parsed matches and run through existing predict_text
    let synthetic: String = matches
        .iter()
        .map(|m| format!("{} vs {} - {}", m.player_a, m.player_b, m.surface))
        .collect::<Vec<_>>()
        .join("\n");

    let elo = match tour.as_str() {
        "wta" => app_state.elo_wta.lock().unwrap(),
        _ => app_state.elo_atp.lock().unwrap(),
    };
    match &*elo {
        None => Err(format!("Elo data for {} not loaded", tour.to_uppercase())),
        Some(state) => Ok(predict_text(&synthetic, state)),
    }
}

pub(crate) fn normalize_player_id(name: &str) -> String {
    name.replace(' ', "_").to_lowercase()
}

fn resolve_player(
    state: &serde_json::Value,
    player_id: &str,
    surface: &str,
) -> Result<(f64, f64), String> {
    let elo_overall = get_player_elo(state, player_id, "overall")
        .or_else(|_| get_player_elo(state, player_id, ""))?;
    let matches_on_surface = get_player_surface_matches(state, player_id, surface).unwrap_or(0);
    let elo_surface = get_player_elo(state, player_id, surface).unwrap_or(elo_overall);
    Ok((elo_overall, surface_elo(elo_surface, elo_overall, matches_on_surface)))
}

fn predict_match(m: &ParsedMatch, state: &serde_json::Value) -> Result<PredictionResult, String> {
    let id_a = normalize_player_id(&m.player_a);
    let id_b = normalize_player_id(&m.player_b);
    let (elo_a_overall, elo_a_composite) = resolve_player(state, &id_a, &m.surface)?;
    let (elo_b_overall, elo_b_composite) = resolve_player(state, &id_b, &m.surface)?;
    let prob_a = expected_probability(elo_a_composite, elo_b_composite);
    Ok(PredictionResult {
        player_a: m.player_a.clone(),
        player_b: m.player_b.clone(),
        surface: m.surface.clone(),
        prob_a_wins: prob_a,
        prob_b_wins: 1.0 - prob_a,
        elo_a_overall,
        elo_b_overall,
        ml_prob_a_wins: None,
        confidence_flag: None,
    })
}

pub fn calculate_kelly_impl(req: KellyRequest) -> Result<KellyResult, String> {
    let implied_prob = kelly::implied_probability(req.decimal_odds);
    let edge = kelly::edge(req.model_prob, req.decimal_odds);
    let full_kelly = kelly::full_kelly_fraction(req.model_prob, req.decimal_odds);
    let fractional_kelly = kelly::fractional_kelly(req.model_prob, req.decimal_odds, req.kelly_fraction);
    let stake = kelly::stake_from_kelly(req.bankroll, fractional_kelly);
    Ok(KellyResult { implied_prob, edge, full_kelly, fractional_kelly, stake })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn elo_state_two_players() -> serde_json::Value {
        json!({
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
        })
    }

    #[test]
    fn test_predict_returns_result() {
        let state = elo_state_two_players();
        let response = predict_text("Alcaraz vs Sinner - Clay", &state);
        assert_eq!(response.predictions.len(), 1);
        assert_eq!(response.data_as_of, "2026-04-20");
        assert!(response.error.is_none());
        assert!(response.predictions[0].prob_a_wins > 0.5);
    }

    #[test]
    fn test_predict_missing_player() {
        let state = json!({"data_as_of": "2026-04-20", "players": {}});
        let response = predict_text("Unknown vs Sinner", &state);
        assert_eq!(response.predictions.len(), 0);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_predict_multiple() {
        let state = json!({
            "data_as_of": "2026-04-20",
            "players": {
                "alcaraz": { "elo_overall": 1600, "elo_hard": 1600, "elo_clay": 1600, "elo_grass": 1600, "matches_played": 10, "matches_played_hard": 5, "matches_played_clay": 5, "matches_played_grass": 0 },
                "sinner": { "elo_overall": 1500, "elo_hard": 1500, "elo_clay": 1500, "elo_grass": 1500, "matches_played": 8, "matches_played_hard": 4, "matches_played_clay": 4, "matches_played_grass": 0 },
                "djokovic": { "elo_overall": 1700, "elo_hard": 1700, "elo_clay": 1700, "elo_grass": 1700, "matches_played": 20, "matches_played_hard": 10, "matches_played_clay": 5, "matches_played_grass": 5 },
                "medvedev": { "elo_overall": 1400, "elo_hard": 1400, "elo_clay": 1400, "elo_grass": 1400, "matches_played": 8, "matches_played_hard": 4, "matches_played_clay": 2, "matches_played_grass": 2 }
            }
        });
        let response = predict_text("Alcaraz vs Sinner\nDjokovic vs Medvedev", &state);
        assert_eq!(response.predictions.len(), 2);
        assert_eq!(response.predictions[0].player_a, "Alcaraz");
        assert_eq!(response.predictions[1].player_a, "Djokovic");
    }

    #[test]
    fn test_predict_empty_input() {
        let state = json!({"data_as_of": "2026-04-20", "players": {}});
        let response = predict_text("", &state);
        assert_eq!(response.predictions.len(), 0);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_predict_probability_bounds() {
        let state = elo_state_two_players();
        let response = predict_text("Alcaraz vs Sinner", &state);
        assert_eq!(response.predictions.len(), 1);
        let pred = &response.predictions[0];
        assert!(pred.prob_a_wins >= 0.0 && pred.prob_a_wins <= 1.0);
        assert!(pred.prob_b_wins >= 0.0 && pred.prob_b_wins <= 1.0);
        assert!((pred.prob_a_wins + pred.prob_b_wins - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_data_as_of_returned_in_response() {
        let state = elo_state_two_players();
        let response = predict_text("Alcaraz vs Sinner", &state);
        assert_eq!(response.data_as_of, "2026-04-20");
    }

    #[test]
    fn test_kelly_request_struct() {
        let req = KellyRequest {
            model_prob: 0.6,
            decimal_odds: 2.0,
            bankroll: 1000.0,
            kelly_fraction: 0.25,
        };
        assert_eq!(req.model_prob, 0.6);
    }

    #[test]
    fn test_calculate_kelly_response() {
        let req = KellyRequest {
            model_prob: 0.6,
            decimal_odds: 2.0,
            bankroll: 1000.0,
            kelly_fraction: 0.25,
        };
        let result = calculate_kelly_impl(req).unwrap();
        assert!((result.edge - 0.1).abs() < 0.001);
        assert!((result.implied_prob - 0.5).abs() < 0.001);
        assert!((result.stake - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_calculate_kelly_negative_edge() {
        let req = KellyRequest {
            model_prob: 0.3,
            decimal_odds: 2.0,
            bankroll: 1000.0,
            kelly_fraction: 0.25,
        };
        let result = calculate_kelly_impl(req).unwrap();
        assert_eq!(result.stake, 0.0);
    }

    fn elo_state_integration() -> serde_json::Value {
        json!({
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
        })
    }

    #[test]
    fn test_scenario_1_basic_kelly() {
        let state = elo_state_integration();
        let response = predict_text("Alcaraz vs Sinner - Hard", &state);
        assert_eq!(response.predictions.len(), 1);
        assert!(response.error.is_none());
        let pred = &response.predictions[0];
        assert_eq!(pred.player_a, "Alcaraz");
        assert_eq!(pred.surface, "Hard");
        assert!(pred.prob_a_wins > 0.5);

        let result = calculate_kelly_impl(KellyRequest {
            model_prob: pred.prob_a_wins,
            decimal_odds: 2.50,
            bankroll: 1000.0,
            kelly_fraction: 0.25,
        }).unwrap();
        assert!((result.implied_prob - 0.4).abs() < 0.001);
        assert!(result.edge > 0.0);
        assert!(result.stake > 0.0);
    }

    #[test]
    fn test_scenario_2_negative_edge() {
        let result = calculate_kelly_impl(KellyRequest {
            model_prob: 0.6,
            decimal_odds: 1.20,
            bankroll: 1000.0,
            kelly_fraction: 0.25,
        }).unwrap();
        assert!((result.implied_prob - 0.8333).abs() < 0.01);
        assert!(result.edge < 0.0);
        assert_eq!(result.stake, 0.0);
    }

    #[test]
    fn test_scenario_3_bankroll_scaling() {
        let stake_1000 = calculate_kelly_impl(KellyRequest {
            model_prob: 0.6, decimal_odds: 2.50, bankroll: 1000.0, kelly_fraction: 0.25,
        }).unwrap().stake;
        let stake_2000 = calculate_kelly_impl(KellyRequest {
            model_prob: 0.6, decimal_odds: 2.50, bankroll: 2000.0, kelly_fraction: 0.25,
        }).unwrap().stake;
        assert!((stake_2000 - 2.0 * stake_1000).abs() < 0.01);
    }

    #[test]
    fn test_scenario_4_kelly_fraction_scaling() {
        let s01 = calculate_kelly_impl(KellyRequest { model_prob: 0.6, decimal_odds: 2.50, bankroll: 1000.0, kelly_fraction: 0.1 }).unwrap().stake;
        let s025 = calculate_kelly_impl(KellyRequest { model_prob: 0.6, decimal_odds: 2.50, bankroll: 1000.0, kelly_fraction: 0.25 }).unwrap().stake;
        let s05 = calculate_kelly_impl(KellyRequest { model_prob: 0.6, decimal_odds: 2.50, bankroll: 1000.0, kelly_fraction: 0.5 }).unwrap().stake;
        let s1 = calculate_kelly_impl(KellyRequest { model_prob: 0.6, decimal_odds: 2.50, bankroll: 1000.0, kelly_fraction: 1.0 }).unwrap().stake;
        assert!((s025 - 2.5 * s01).abs() < 0.01);
        assert!((s05 - 2.0 * s025).abs() < 0.01);
        assert!((s1 - 2.0 * s05).abs() < 0.01);
    }

    #[test]
    fn test_scenario_5_edge_display() {
        let pos = calculate_kelly_impl(KellyRequest { model_prob: 0.6, decimal_odds: 2.50, bankroll: 1000.0, kelly_fraction: 0.25 }).unwrap();
        assert!(pos.edge > 0.0);
        let neg = calculate_kelly_impl(KellyRequest { model_prob: 0.3, decimal_odds: 2.50, bankroll: 1000.0, kelly_fraction: 0.25 }).unwrap();
        assert!(neg.edge < 0.0);
        let zero = calculate_kelly_impl(KellyRequest { model_prob: 0.4, decimal_odds: 2.50, bankroll: 1000.0, kelly_fraction: 0.25 }).unwrap();
        assert!(zero.edge.abs() < 0.001);
    }

    #[test]
    fn test_scenario_6_multiple_matches() {
        let state = elo_state_integration();
        let response = predict_text("Alcaraz vs Sinner - Hard\nDjokovic vs Medvedev - Hard", &state);
        assert_eq!(response.predictions.len(), 2);
        assert!(response.error.is_none());
        let r1 = calculate_kelly_impl(KellyRequest { model_prob: response.predictions[0].prob_a_wins, decimal_odds: 2.50, bankroll: 1000.0, kelly_fraction: 0.25 }).unwrap();
        let r2 = calculate_kelly_impl(KellyRequest { model_prob: response.predictions[1].prob_a_wins, decimal_odds: 3.00, bankroll: 1000.0, kelly_fraction: 0.25 }).unwrap();
        assert!(r1.stake >= 0.0);
        assert!(r2.stake >= 0.0);
    }

    #[test]
    fn test_scenario_7_empty_input() {
        let state = json!({"data_as_of": "2026-04-20", "players": {}});
        let response = predict_text("", &state);
        assert_eq!(response.predictions.len(), 0);
        assert!(response.error.is_some());
    }
}
