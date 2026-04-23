use serde::{Deserialize, Serialize};
use crate::artifacts::{get_data_as_of, get_player_elo, get_player_surface_matches};
use crate::elo::{expected_probability, surface_elo};
use crate::parser::{parse_match_text, ParsedMatch};

#[derive(Serialize, Deserialize, Clone)]
pub struct PredictionResult {
    pub player_a: String,
    pub player_b: String,
    pub surface: String,
    pub prob_a_wins: f64,
    pub prob_b_wins: f64,
    pub elo_a_overall: f64,
    pub elo_b_overall: f64,
}

#[derive(Serialize, Deserialize)]
pub struct PredictResponse {
    pub predictions: Vec<PredictionResult>,
    pub data_as_of: String,
    pub error: Option<String>,
}

/// Parse match text and compute Elo predictions.
/// Accepts elo_json as JSON string from the UI.
#[tauri::command]
pub fn parse_and_predict(text: String, elo_json: String) -> PredictResponse {
    let matches = match parse_match_text(&text) {
        Ok(m) => m,
        Err(e) => {
            return PredictResponse {
                predictions: vec![],
                data_as_of: "unknown".to_string(),
                error: Some(e),
            }
        }
    };

    let state = match serde_json::from_str(&elo_json) {
        Ok(s) => s,
        Err(e) => {
            return PredictResponse {
                predictions: vec![],
                data_as_of: "unknown".to_string(),
                error: Some(format!("Invalid Elo JSON: {}", e)),
            }
        }
    };

    let data_as_of = get_data_as_of(&state);
    let mut predictions = Vec::new();

    for m in matches {
        match predict_match(&m, &state) {
            Ok(pred) => predictions.push(pred),
            Err(_e) => {
                // Skip unparseable matches silently, allow partial results
            }
        }
    }

    PredictResponse {
        predictions,
        data_as_of,
        error: if predictions.is_empty() {
            Some("No matches could be predicted".to_string())
        } else {
            None
        },
    }
}

fn predict_match(m: &ParsedMatch, state: &serde_json::Value) -> Result<PredictionResult, String> {
    let player_id_a = m.player_a.replace(" ", "_").to_lowercase();
    let player_id_b = m.player_b.replace(" ", "_").to_lowercase();

    let elo_a_overall = get_player_elo(state, &player_id_a, "overall")
        .or_else(|_| get_player_elo(state, &player_id_a, ""))?;
    let elo_b_overall = get_player_elo(state, &player_id_b, "overall")
        .or_else(|_| get_player_elo(state, &player_id_b, ""))?;

    let matches_a = get_player_surface_matches(state, &player_id_a, &m.surface).unwrap_or(0);
    let matches_b = get_player_surface_matches(state, &player_id_b, &m.surface).unwrap_or(0);

    let elo_a_surface = get_player_elo(state, &player_id_a, &m.surface).unwrap_or(elo_a_overall);
    let elo_b_surface = get_player_elo(state, &player_id_b, &m.surface).unwrap_or(elo_b_overall);

    let elo_a_composite = surface_elo(elo_a_surface, elo_a_overall, matches_a);
    let elo_b_composite = surface_elo(elo_b_surface, elo_b_overall, matches_b);

    let prob_a = expected_probability(elo_a_composite, elo_b_composite);
    let prob_b = 1.0 - prob_a;

    Ok(PredictionResult {
        player_a: m.player_a.clone(),
        player_b: m.player_b.clone(),
        surface: m.surface.clone(),
        prob_a_wins: prob_a,
        prob_b_wins: prob_b,
        elo_a_overall,
        elo_b_overall,
    })
}

/// Get the data-as-of date for the Elo model.
#[tauri::command]
pub fn get_data_as_of_cmd(elo_json: String) -> String {
    serde_json::from_str::<serde_json::Value>(&elo_json)
        .ok()
        .and_then(|state| {
            state
                .get("data_as_of")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}
