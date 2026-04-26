/// RapidAPI Tennis (tennisapi1.p.rapidapi.com) — SofaScore-backed.
///
/// Working endpoints (confirmed):
///   GET /api/tennis/rankings/atp
///   GET /api/tennis/rankings/wta
///   GET /api/tennis/team/{id}/events/previous/0  (recent + scheduled events)
///   GET /api/tennis/team/{id}/events/next/0      (future events; often 204)
///
/// Schedule strategy: fetch rankings → parallel player event requests → filter by date+status.
use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::schedule::{ScheduleMatch, ScheduledPrediction, surface_for, level_and_best_of, round_from_str};

const BASE: &str = "https://tennisapi1.p.rapidapi.com/api/tennis";
const HOST: &str = "tennisapi1.p.rapidapi.com";

// ── Response types ───────────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct EventsResponse {
    #[serde(default)]
    events: Vec<Event>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct Event {
    home_team: Team,
    away_team: Team,
    tournament: TournamentInfo,
    #[serde(default)]
    round_info: Option<RoundInfo>,
    status: StatusInfo,
    #[serde(default)]
    start_timestamp: Option<i64>,
    /// Top-level groundType field: "Red clay" | "Grass" | "Hard" | "Hard (Indoor)"
    #[serde(default)]
    ground_type: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
struct Team {
    name: String,
    #[serde(default)]
    id: Option<u32>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct TournamentInfo {
    #[serde(default)]
    name: String,
    #[serde(default)]
    unique_tournament: Option<UniqueTournament>,
    #[serde(default)]
    category: Option<Category>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct UniqueTournament {
    #[serde(default)]
    name: Option<String>,
    /// "Red clay" | "Grass" | "Hard" | "Hard (Indoor)"
    #[serde(default)]
    ground_type: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
struct Category {
    #[serde(default)]
    name: Option<String>,
    /// ATP = 3, WTA = 6
    #[serde(default)]
    id: Option<u32>,
}

#[derive(Deserialize, Debug, Clone)]
struct RoundInfo {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    round: Option<u32>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct StatusInfo {
    #[serde(rename = "type")]
    status_type: String,
}

// ── Rankings ─────────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct RankingsResponse {
    rankings: Vec<RankingRow>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct RankingRow {
    ranking: u32,
    team: Team,
    /// Confirmed field name from live API
    #[serde(default)]
    points: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RankingEntry {
    pub rank: u32,
    pub name: String,
    pub points: u32,
    pub player_id: Option<u32>,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn rapidapi_get(url: &str, api_key: &str) -> reqwest::RequestBuilder {
    Client::new()
        .get(url)
        .header("x-rapidapi-host", HOST)
        .header("x-rapidapi-key", api_key)
}

fn is_atp(cat: &Option<Category>) -> bool {
    cat.as_ref().map_or(false, |c| {
        c.id == Some(3)
            || c.name.as_deref().map(|n| n.eq_ignore_ascii_case("atp")).unwrap_or(false)
    })
}

fn is_wta(cat: &Option<Category>) -> bool {
    cat.as_ref().map_or(false, |c| {
        c.id == Some(6)
            || c.name.as_deref().map(|n| n.eq_ignore_ascii_case("wta")).unwrap_or(false)
    })
}

fn is_not_started(s: &StatusInfo) -> bool {
    matches!(s.status_type.as_str(), "notstarted" | "not_started" | "scheduled")
}

/// "Jannik Sinner" → "Sinner",  "Carlos Alcaraz" → "Alcaraz"
/// SofaScore uses "First Last" format.
fn last_name_from_full(name: &str) -> String {
    name.split_whitespace().last().unwrap_or(name).to_string()
}

fn surface_from_event(ev: &Event) -> &'static str {
    // Try top-level groundType first (confirmed from live API)
    let from_ground = |g: &str| -> &'static str {
        let l = g.to_lowercase();
        if l.contains("clay") { "Clay" }
        else if l.contains("grass") { "Grass" }
        else { "Hard" }
    };

    if let Some(g) = ev.ground_type.as_deref() {
        return from_ground(g);
    }
    if let Some(ut) = ev.tournament.unique_tournament.as_ref() {
        if let Some(g) = ut.ground_type.as_deref() {
            return from_ground(g);
        }
        if let Some(name) = ut.name.as_deref() {
            return surface_for(name);
        }
    }
    surface_for(&ev.tournament.name)
}

fn tournament_name(ev: &Event) -> String {
    ev.tournament
        .unique_tournament
        .as_ref()
        .and_then(|u| u.name.clone())
        .unwrap_or_else(|| ev.tournament.name.clone())
}

fn round_name(ev: &Event) -> String {
    ev.round_info
        .as_ref()
        .and_then(|r| r.name.as_deref())
        .map(round_from_str)
        .map(str::to_string)
        .or_else(|| ev.round_info.as_ref().and_then(|r| r.round.map(|n| format!("R{n}"))))
        .unwrap_or_else(|| "R32".to_string())
}

fn event_time_utc(ts: Option<i64>) -> String {
    match ts {
        Some(t) => {
            let h = (t % 86400) / 3600;
            let m = (t % 3600) / 60;
            format!("{:02}:{:02}", h, m)
        }
        None => String::new(),
    }
}

/// True if the event's startTimestamp falls within the given date (UTC).
fn on_date(ts: Option<i64>, date: &str) -> bool {
    let Some(t) = ts else { return true }; // if no timestamp, include it
    // Parse date → epoch range [start, start + 86400)
    let Ok(d) = chrono_day_start(date) else { return true };
    t >= d && t < d + 86400
}

fn chrono_day_start(date: &str) -> Result<i64> {
    // date = "YYYY-MM-DD"
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 { return Err(anyhow!("bad date")); }
    let y: i64 = parts[0].parse()?;
    let m: i64 = parts[1].parse()?;
    let d: i64 = parts[2].parse()?;
    // days since epoch (rough, ignores leap seconds)
    let days = days_from_civil(y, m, d);
    Ok(days * 86400)
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn event_to_schedule_match(ev: &Event, date: &str) -> ScheduleMatch {
    let t = tournament_name(ev);
    let surface = surface_from_event(ev).to_string();
    let (tourney_level, best_of) = level_and_best_of(&t);
    ScheduleMatch {
        player_a_full: ev.home_team.name.clone(),
        player_b_full: ev.away_team.name.clone(),
        player_a_last: last_name_from_full(&ev.home_team.name),
        player_b_last: last_name_from_full(&ev.away_team.name),
        surface,
        tournament: t,
        tourney_level: tourney_level.to_string(),
        round: round_name(ev),
        best_of,
        event_time: event_time_utc(ev.start_timestamp),
        event_date: date.to_string(),
    }
}

// ── Public fetching functions ─────────────────────────────────────────────────

/// Fetch ATP or WTA rankings (top 50).
pub async fn fetch_rankings_list(api_key: &str, tour: &str) -> Result<Vec<RankingEntry>> {
    let url = format!("{BASE}/rankings/{tour}");
    let resp = rapidapi_get(&url, api_key)
        .send()
        .await?
        .error_for_status()?
        .json::<RankingsResponse>()
        .await
        .map_err(|e| anyhow!("Rankings parse error: {e}"))?;

    Ok(resp
        .rankings
        .into_iter()
        .map(|r| RankingEntry {
            rank: r.ranking,
            name: r.team.name,
            points: r.points,
            player_id: r.team.id,
        })
        .collect())
}

/// Fetch today's scheduled matches via player event lookups.
/// Strategy: get top-N rankings → fetch each player's recent+upcoming events in parallel
/// → filter by date & notstarted & tour → deduplicate.
pub async fn fetch_schedule_via_players(
    api_key: &str,
    tour: &str,
    date: &str,
    top_n: usize,
) -> Result<Vec<ScheduleMatch>> {
    // Step 1: get rankings to collect player IDs
    let rankings = fetch_rankings_list(api_key, tour).await?;
    let player_ids: Vec<u32> = rankings
        .into_iter()
        .filter_map(|r| r.player_id)
        .take(top_n)
        .collect();

    // Step 2: fetch each player's events in parallel
    let key = api_key.to_string();
    let date_str = date.to_string();
    let tour_str = tour.to_string();

    let futures: Vec<_> = player_ids
        .into_iter()
        .map(|id| {
            let k = key.clone();
            let d = date_str.clone();
            let t = tour_str.clone();
            async move { fetch_player_events_for_date(&k, id, &d, &t).await }
        })
        .collect();

    let results = futures::future::join_all(futures).await;

    // Step 3: merge and deduplicate by sorted player pair
    let mut seen: HashSet<String> = HashSet::new();
    let mut matches: Vec<ScheduleMatch> = Vec::new();

    for result in results.into_iter().flatten().flatten() {
        let key = {
            let mut names = vec![result.player_a_last.to_lowercase(), result.player_b_last.to_lowercase()];
            names.sort();
            names.join("|")
        };
        if seen.insert(key) {
            matches.push(result);
        }
    }

    matches.sort_by(|a, b| a.event_time.cmp(&b.event_time));
    Ok(matches)
}

async fn fetch_player_events_for_date(
    api_key: &str,
    player_id: u32,
    date: &str,
    tour: &str,
) -> Result<Vec<ScheduleMatch>> {
    let url = format!("{BASE}/team/{player_id}/events/previous/0");
    let resp = rapidapi_get(&url, api_key)
        .send()
        .await?;

    if resp.status() == reqwest::StatusCode::NO_CONTENT {
        return Ok(vec![]);
    }

    let body = resp.error_for_status()?.json::<EventsResponse>().await?;

    let is_target_tour = |ev: &Event| match tour {
        "wta" => is_wta(&ev.tournament.category),
        _ => is_atp(&ev.tournament.category),
    };

    Ok(body
        .events
        .into_iter()
        .filter(|ev| is_not_started(&ev.status) && is_target_tour(ev) && on_date(ev.start_timestamp, date))
        .map(|ev| event_to_schedule_match(&ev, date))
        .collect())
}

// ── Tauri commands ────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn fetch_rankings(
    api_key: String,
    tour: String,
) -> Result<Vec<RankingEntry>, String> {
    fetch_rankings_list(&api_key, &tour)
        .await
        .map_err(|e| e.to_string())
}

/// Fetch schedule using RapidAPI with Elo/ML enrichment.
#[tauri::command]
pub async fn fetch_rapidapi_schedule(
    api_key: String,
    tour: String,
    date: String,
    app_state: tauri::State<'_, crate::state::AppState>,
    sidecar_state: tauri::State<'_, std::sync::Mutex<crate::sidecar::SidecarState>>,
) -> Result<Vec<ScheduledPrediction>, String> {
    let matches = fetch_schedule_via_players(&api_key, &tour, &date, 30)
        .await
        .map_err(|e| e.to_string())?;

    enrich_matches(matches, &tour, &date, app_state, sidecar_state).await
}

/// Try api-tennis.com first; fall back to RapidAPI if it fails or returns nothing.
#[tauri::command]
pub async fn fetch_schedule_auto(
    api_tennis_key: String,
    rapid_key: String,
    tour: String,
    date: String,
    app_state: tauri::State<'_, crate::state::AppState>,
    sidecar_state: tauri::State<'_, std::sync::Mutex<crate::sidecar::SidecarState>>,
) -> Result<(Vec<ScheduledPrediction>, String), String> {
    // Primary: api-tennis.com
    if !api_tennis_key.is_empty() {
        match crate::schedule::fetch_schedule(&api_tennis_key, &tour, &date).await {
            Ok(matches) if !matches.is_empty() => {
                let preds = enrich_matches(matches, &tour, &date, app_state, sidecar_state).await?;
                return Ok((preds, "api-tennis.com".to_string()));
            }
            Ok(_) => eprintln!("[schedule] api-tennis.com returned 0 matches, trying RapidAPI"),
            Err(e) => eprintln!("[schedule] api-tennis.com failed: {e}, trying RapidAPI"),
        }
    }

    // Fallback: RapidAPI
    if rapid_key.is_empty() {
        return Err("No matches found. Both API keys are needed for fallback.".to_string());
    }
    let matches = fetch_schedule_via_players(&rapid_key, &tour, &date, 30)
        .await
        .map_err(|e| format!("RapidAPI fallback failed: {e}"))?;

    let preds = enrich_matches(matches, &tour, &date, app_state, sidecar_state).await?;
    Ok((preds, "RapidAPI".to_string()))
}

async fn enrich_matches(
    matches: Vec<ScheduleMatch>,
    tour: &str,
    date: &str,
    app_state: tauri::State<'_, crate::state::AppState>,
    sidecar_state: tauri::State<'_, std::sync::Mutex<crate::sidecar::SidecarState>>,
) -> Result<Vec<ScheduledPrediction>, String> {
    use crate::commands::{predict_text, normalize_player_id, PredictResponse};
    use crate::sidecar::{ml_predict, MlMatchRequest};

    if matches.is_empty() {
        return Ok(vec![]);
    }

    let synthetic: String = matches
        .iter()
        .map(|m| format!("{} vs {} - {}", m.player_a_last, m.player_b_last, m.surface))
        .collect::<Vec<_>>()
        .join("\n");

    let elo_resp: PredictResponse = {
        let guard = match tour {
            "wta" => app_state.elo_wta.lock().unwrap(),
            _ => app_state.elo_atp.lock().unwrap(),
        };
        match &*guard {
            None => return Err(format!("Elo data for {} not loaded.", tour.to_uppercase())),
            Some(state) => predict_text(&synthetic, state),
        }
    };

    let port = sidecar_state.lock().unwrap().port;
    let ml_map: Vec<Option<(f64, String)>> = if let Some(p) = port {
        let reqs: Vec<MlMatchRequest> = elo_resp.predictions.iter().zip(matches.iter()).map(|(pred, m)| {
            MlMatchRequest {
                tour: tour.to_string(),
                player_a_id: normalize_player_id(&pred.player_a),
                player_b_id: normalize_player_id(&pred.player_b),
                surface: m.surface.clone(),
                tourney_level: m.tourney_level.clone(),
                round_: m.round.clone(),
                best_of: m.best_of,
                tourney_date: date.to_string(),
            }
        }).collect();

        match ml_predict(p, reqs).await {
            Ok(r) => r.predictions.into_iter().map(|p| Some((p.prob_a_wins, p.confidence_flag))).collect(),
            Err(_) => vec![None; elo_resp.predictions.len()],
        }
    } else {
        vec![None; elo_resp.predictions.len()]
    };

    Ok(elo_resp.predictions.into_iter()
        .zip(matches.iter())
        .zip(ml_map)
        .map(|((pred, m), ml)| ScheduledPrediction {
            player_a: m.player_a_full.clone(),
            player_b: m.player_b_full.clone(),
            surface: m.surface.clone(),
            prob_a_wins: pred.prob_a_wins,
            prob_b_wins: pred.prob_b_wins,
            elo_a_overall: pred.elo_a_overall,
            elo_b_overall: pred.elo_b_overall,
            ml_prob_a_wins: ml.as_ref().map(|(p, _)| *p),
            confidence_flag: ml.map(|(_, f)| f),
            player_a_full: m.player_a_full.clone(),
            player_b_full: m.player_b_full.clone(),
            tournament: m.tournament.clone(),
            round: m.round.clone(),
            event_time: m.event_time.clone(),
            event_date: m.event_date.clone(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_last_name_first_last_format() {
        // SofaScore uses "First Last" confirmed from live API
        assert_eq!(last_name_from_full("Jannik Sinner"), "Sinner");
        assert_eq!(last_name_from_full("Carlos Alcaraz"), "Alcaraz");
        assert_eq!(last_name_from_full("Alexander Zverev"), "Zverev");
    }

    #[test]
    fn test_last_name_single_word() {
        assert_eq!(last_name_from_full("Sinner"), "Sinner");
    }

    #[test]
    fn test_is_not_started() {
        let s = |t: &str| StatusInfo { status_type: t.to_string() };
        assert!(is_not_started(&s("notstarted")));
        assert!(is_not_started(&s("scheduled")));
        assert!(!is_not_started(&s("inprogress")));
        assert!(!is_not_started(&s("finished")));
    }

    #[test]
    fn test_surface_from_ground_type_field() {
        fn make_ev(ground_type: &str) -> Event {
            Event {
                home_team: Team { name: "A".into(), id: None },
                away_team: Team { name: "B".into(), id: None },
                tournament: TournamentInfo { name: "Test".into(), unique_tournament: None, category: None },
                round_info: None,
                status: StatusInfo { status_type: "notstarted".into() },
                start_timestamp: None,
                ground_type: Some(ground_type.into()),
            }
        }
        assert_eq!(surface_from_event(&make_ev("Red clay")), "Clay");
        assert_eq!(surface_from_event(&make_ev("Grass")), "Grass");
        assert_eq!(surface_from_event(&make_ev("Hard (Indoor)")), "Hard");
        assert_eq!(surface_from_event(&make_ev("Hard")), "Hard");
    }

    #[test]
    fn test_atp_category() {
        let cat = Some(Category { id: Some(3), name: Some("ATP".into()) });
        assert!(is_atp(&cat));
        assert!(!is_wta(&cat));
    }

    #[test]
    fn test_wta_category() {
        let cat = Some(Category { id: Some(6), name: Some("WTA".into()) });
        assert!(is_wta(&cat));
        assert!(!is_atp(&cat));
    }

    #[test]
    fn test_chrono_day_start_known_date() {
        // 2026-01-01 = 20454 days since epoch → 20454 * 86400
        let t = chrono_day_start("2026-04-25").unwrap();
        assert!(t > 0);
        // Any 2026-04-25 event timestamp should fall in [t, t+86400)
        // 2026-04-25 00:00 UTC = 1777075200
        assert!((t - 1777075200).abs() < 86400);
    }

    #[test]
    fn test_on_date_within_range() {
        let t = chrono_day_start("2026-04-25").unwrap();
        assert!(on_date(Some(t + 3600), "2026-04-25")); // 01:00 UTC on that day
        assert!(!on_date(Some(t - 1), "2026-04-25")); // second before midnight
        assert!(!on_date(Some(t + 86400), "2026-04-25")); // next day
    }
}
