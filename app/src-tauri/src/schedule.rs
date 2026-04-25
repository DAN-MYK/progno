use std::collections::HashMap;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct ScheduleMatch {
    pub player_a_full: String,
    pub player_b_full: String,
    pub player_a_last: String,
    pub player_b_last: String,
    pub surface: String,
    pub tournament: String,
    pub tourney_level: String,
    pub round: String,
    pub best_of: u8,
    pub event_time: String,
    pub event_date: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ScheduledPrediction {
    pub player_a: String,
    pub player_b: String,
    pub surface: String,
    pub prob_a_wins: f64,
    pub prob_b_wins: f64,
    pub elo_a_overall: f64,
    pub elo_b_overall: f64,
    pub ml_prob_a_wins: Option<f64>,
    pub confidence_flag: Option<String>,
    pub player_a_full: String,
    pub player_b_full: String,
    pub tournament: String,
    pub round: String,
    pub event_time: String,
    pub event_date: String,
}

fn player_last_name(api_name: &str) -> String {
    api_name.split_whitespace().last().unwrap_or(api_name).to_string()
}

pub fn round_from_str(s: &str) -> &'static str {
    let l = s.to_lowercase();
    if l.contains("1/32") { return "R32"; }
    if l.contains("1/16") { return "R16"; }
    if l.contains("1/8") || l.contains("quarterfinal") { return "QF"; }
    if l.contains("1/4") || l.contains("semifinal") { return "SF"; }
    if l.contains("final") { return "F"; }
    "R32"
}

pub fn surface_for(tournament: &str) -> &'static str {
    let l = tournament.to_lowercase();
    let clay = ["madrid", "rome", "italian", "roland garros", "french", "monte carlo",
                "monte-carlo", "barcelona", "hamburg", "geneva", "lyon", "estoril",
                "bucharest", "munich", "marrakech", "marrakesh", "belgrade",
                "buenos aires", "houston", "cordoba", "rio", "santiago", "bastad",
                "kitzbuhel", "gstaad", "umag", "palermo", "lausanne", "istanbul",
                "casablanca"];
    let grass = ["wimbledon", "queens", "queen's", "halle", "eastbourne",
                 "hertogenbosch", "birmingham", "nottingham", "mallorca"];
    if clay.iter().any(|k| l.contains(k)) { return "Clay"; }
    if grass.iter().any(|k| l.contains(k)) { return "Grass"; }
    "Hard"
}

pub fn level_and_best_of(tournament: &str) -> (&'static str, u8) {
    let l = tournament.to_lowercase();
    let gs = ["australian open", "french open", "roland garros", "wimbledon", "us open"];
    let m1000 = ["indian wells", "miami", "monte carlo", "monte-carlo", "madrid",
                 "rome", "italian", "canada", "montreal", "toronto", "cincinnati",
                 "western", "shanghai", "paris masters", "paris"];
    if gs.iter().any(|k| l.contains(k)) { return ("G", 5); }
    if m1000.iter().any(|k| l.contains(k)) { return ("M", 3); }
    ("A", 3)
}

fn is_upcoming(status: &str) -> bool {
    matches!(status.trim(), "1" | "" | "NS" | "Not Started" | "Scheduled")
}

fn str_val<'a>(v: &'a serde_json::Value, key: &str) -> &'a str {
    v.get(key).and_then(|x| x.as_str()).unwrap_or("")
}

pub async fn fetch_schedule(
    api_key: &str,
    tour: &str,
    date: &str,
) -> anyhow::Result<Vec<ScheduleMatch>> {
    let event_type = if tour == "wta" { "Wta Singles" } else { "Atp Singles" };

    let url = format!(
        "https://api.api-tennis.com/tennis/?method=get_fixtures&APIkey={}&date_start={}&date_stop={}",
        api_key, date, date
    );

    let body: serde_json::Value = reqwest::get(&url).await?.json().await?;

    if body["success"].as_u64() != Some(1) {
        let msg = body["error"].as_str().unwrap_or("API request failed");
        return Err(anyhow::anyhow!("{}", msg));
    }

    let items = body["result"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Unexpected API response format"))?;

    let matches = items
        .iter()
        .filter(|f| {
            str_val(f, "event_type_type") == event_type
                && str_val(f, "event_qualification") != "True"
                && is_upcoming(str_val(f, "event_status"))
                && !str_val(f, "event_first_player").is_empty()
                && !str_val(f, "event_second_player").is_empty()
        })
        .map(|f| {
            let p1 = str_val(f, "event_first_player").to_string();
            let p2 = str_val(f, "event_second_player").to_string();
            let tournament = str_val(f, "tournament_name").to_string();
            let round = round_from_str(str_val(f, "tournament_round")).to_string();
            let surface = surface_for(&tournament).to_string();
            let (level, best_of) = level_and_best_of(&tournament);
            ScheduleMatch {
                player_a_last: player_last_name(&p1),
                player_b_last: player_last_name(&p2),
                player_a_full: p1,
                player_b_full: p2,
                surface,
                tournament,
                tourney_level: level.to_string(),
                round,
                best_of,
                event_time: str_val(f, "event_time").to_string(),
                event_date: str_val(f, "event_date").to_string(),
            }
        })
        .collect();

    Ok(matches)
}

#[cfg(not(test))]
#[tauri::command]
pub async fn fetch_and_predict(
    api_key: String,
    tour: String,
    date: String,
    app_state: tauri::State<'_, crate::state::AppState>,
    sidecar_state: tauri::State<'_, std::sync::Mutex<crate::sidecar::SidecarState>>,
) -> Result<Vec<ScheduledPrediction>, String> {
    use crate::commands::predict_text;
    use crate::sidecar::{MlMatchRequest, ml_predict};

    let schedule = fetch_schedule(&api_key, &tour, &date)
        .await
        .map_err(|e| e.to_string())?;

    if schedule.is_empty() {
        return Ok(vec![]);
    }

    let elo_state = {
        let guard = match tour.as_str() {
            "wta" => app_state.elo_wta.lock().unwrap(),
            _ => app_state.elo_atp.lock().unwrap(),
        };
        match &*guard {
            None => return Err(format!(
                "Elo data for {} not loaded. Run 'just elo' first.",
                tour.to_uppercase()
            )),
            Some(elo) => elo.clone(),
        }
    };

    // Elo prediction per match (last-name lookup)
    let mut elo_preds: Vec<Option<crate::commands::PredictionResult>> = Vec::new();
    for m in &schedule {
        let text = format!("{} vs {} - {}", m.player_a_last, m.player_b_last, m.surface);
        let resp = predict_text(&text, &elo_state);
        elo_preds.push(resp.predictions.into_iter().next());
    }

    // Batch ML for successful Elo predictions
    let port = sidecar_state.lock().unwrap().port;
    let mut ml_map: HashMap<usize, (f64, String)> = HashMap::new();

    if let Some(port) = port {
        let indexed: Vec<(usize, MlMatchRequest)> = schedule
            .iter()
            .enumerate()
            .filter_map(|(i, m)| {
                let pred = elo_preds[i].as_ref()?;
                Some((
                    i,
                    MlMatchRequest {
                        tour: tour.clone(),
                        player_a_id: crate::commands::normalize_player_id(&pred.player_a),
                        player_b_id: crate::commands::normalize_player_id(&pred.player_b),
                        surface: m.surface.clone(),
                        tourney_level: m.tourney_level.clone(),
                        round_: m.round.clone(),
                        best_of: m.best_of,
                        tourney_date: date.clone(),
                    },
                ))
            })
            .collect();

        if !indexed.is_empty() {
            let (idxs, reqs): (Vec<_>, Vec<_>) = indexed.into_iter().unzip();
            if let Ok(ml_resp) = ml_predict(port, reqs).await {
                for (i, ml_pred) in idxs.into_iter().zip(ml_resp.predictions) {
                    ml_map.insert(i, (ml_pred.prob_a_wins, ml_pred.confidence_flag));
                }
            }
        }
    }

    // Combine schedule info + predictions
    let result = schedule
        .into_iter()
        .enumerate()
        .filter_map(|(i, m)| {
            let pred = elo_preds[i].clone()?;
            let (ml_prob, cf) = match ml_map.remove(&i) {
                Some((p, f)) => (Some(p), Some(f)),
                None => (None, None),
            };
            Some(ScheduledPrediction {
                player_a: m.player_a_full.clone(),
                player_b: m.player_b_full.clone(),
                surface: pred.surface,
                prob_a_wins: pred.prob_a_wins,
                prob_b_wins: pred.prob_b_wins,
                elo_a_overall: pred.elo_a_overall,
                elo_b_overall: pred.elo_b_overall,
                ml_prob_a_wins: ml_prob,
                confidence_flag: cf,
                player_a_full: m.player_a_full,
                player_b_full: m.player_b_full,
                tournament: m.tournament,
                round: m.round,
                event_time: m.event_time,
                event_date: m.event_date,
            })
        })
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_last_name_single_word() {
        assert_eq!(player_last_name("Sinner"), "Sinner");
    }

    #[test]
    fn player_last_name_abbreviated() {
        assert_eq!(player_last_name("J. Sinner"), "Sinner");
    }

    #[test]
    fn player_last_name_multi_word() {
        assert_eq!(player_last_name("A. Davidovich Fokina"), "Fokina");
    }

    #[test]
    fn round_mapping() {
        assert_eq!(round_from_str("ATP Madrid - 1/32-finals"), "R32");
        assert_eq!(round_from_str("ATP Madrid - 1/16-finals"), "R16");
        assert_eq!(round_from_str("Quarterfinals"), "QF");
        assert_eq!(round_from_str("Semifinals"), "SF");
        assert_eq!(round_from_str("Final"), "F");
        assert_eq!(round_from_str("unknown"), "R32");
    }

    #[test]
    fn surface_mapping() {
        assert_eq!(surface_for("Madrid"), "Clay");
        assert_eq!(surface_for("Roland Garros"), "Clay");
        assert_eq!(surface_for("Wimbledon"), "Grass");
        assert_eq!(surface_for("Halle"), "Grass");
        assert_eq!(surface_for("Australian Open"), "Hard");
        assert_eq!(surface_for("Miami"), "Hard");
    }

    #[test]
    fn level_mapping() {
        assert_eq!(level_and_best_of("Roland Garros"), ("G", 5));
        assert_eq!(level_and_best_of("Madrid"), ("M", 3));
        assert_eq!(level_and_best_of("Some ATP 250"), ("A", 3));
    }

    #[test]
    fn upcoming_filter() {
        assert!(is_upcoming("1"));
        assert!(is_upcoming(""));
        assert!(!is_upcoming("Finished"));
        assert!(!is_upcoming("Retired"));
        assert!(!is_upcoming("Walk Over"));
    }
}
