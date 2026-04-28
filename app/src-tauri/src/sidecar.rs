use std::sync::Mutex;
use serde::{Deserialize, Serialize};
use tauri::Manager;
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandEvent;

pub struct SidecarState {
    pub port: Option<u16>,
    pub child: Option<tauri_plugin_shell::process::CommandChild>,
}

impl Default for SidecarState {
    fn default() -> Self { Self { port: None, child: None } }
}

#[derive(Serialize)]
pub struct MlMatchRequest {
    pub tour: String,
    pub player_a_id: String,
    pub player_b_id: String,
    pub surface: String,
    pub tourney_level: String,
    pub round_: String,
    pub best_of: u8,
    pub tourney_date: String,
}

#[derive(Serialize)]
struct PredictPayload {
    matches: Vec<MlMatchRequest>,
}

#[derive(Deserialize)]
pub struct MlMatchPrediction {
    pub prob_a_wins: f64,
    pub confidence_flag: String,
}

#[derive(Deserialize)]
pub struct MlPredictResponse {
    pub predictions: Vec<MlMatchPrediction>,
}

pub fn spawn_sidecar(app: &tauri::AppHandle, artifacts_root: String) {
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        match do_spawn(&handle, &artifacts_root).await {
            Ok((port, child)) => {
                let state = handle.state::<Mutex<SidecarState>>();
                let mut guard = state.lock().unwrap();
                guard.port = Some(port);
                guard.child = Some(child);
                eprintln!("[sidecar] ready on port {port}");
            }
            Err(e) => eprintln!("[sidecar] failed to start: {e}"),
        }
    });
}

async fn do_spawn(app: &tauri::AppHandle, artifacts_root: &str) -> anyhow::Result<(u16, tauri_plugin_shell::process::CommandChild)> {
    let (mut rx, child) = app
        .shell()
        .sidecar("progno-sidecar")?
        .args(["--artifacts-root", artifacts_root])
        .spawn()?;

    while let Some(event) = rx.recv().await {
        match event {
            CommandEvent::Stdout(line) => {
                let s = String::from_utf8_lossy(&line);
                if let Some(port_str) = s.trim().strip_prefix("READY port=") {
                    return Ok((port_str.parse()?, child));
                }
            }
            CommandEvent::Stderr(line) => {
                eprintln!("[sidecar stderr] {}", String::from_utf8_lossy(&line));
            }
            CommandEvent::Terminated(status) => {
                return Err(anyhow::anyhow!("sidecar exited: {:?}", status));
            }
            _ => {}
        }
    }
    Err(anyhow::anyhow!("sidecar did not emit READY"))
}

pub async fn ml_predict(port: u16, matches: Vec<MlMatchRequest>) -> anyhow::Result<MlPredictResponse> {
    let url = format!("http://127.0.0.1:{port}/predict");
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&PredictPayload { matches })
        .send()
        .await?
        .error_for_status()?
        .json::<MlPredictResponse>()
        .await?;
    Ok(resp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ml_match_request_serializes_expected_keys() {
        let req = MlMatchRequest {
            tour: "atp".into(),
            player_a_id: "100001".into(),
            player_b_id: "100002".into(),
            surface: "Hard".into(),
            tourney_level: "A".into(),
            round_: "R32".into(),
            best_of: 3,
            tourney_date: "2024-01-01".into(),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["tour"], "atp");
        assert_eq!(json["player_a_id"], "100001");
        assert_eq!(json["player_b_id"], "100002");
        assert_eq!(json["surface"], "Hard");
        assert_eq!(json["tourney_level"], "A");
        assert_eq!(json["round_"], "R32");
        assert_eq!(json["best_of"], 3);
        assert_eq!(json["tourney_date"], "2024-01-01");
    }

    #[test]
    fn predict_payload_wraps_matches_array() {
        let req = MlMatchRequest {
            tour: "wta".into(),
            player_a_id: "200001".into(),
            player_b_id: "200002".into(),
            surface: "Clay".into(),
            tourney_level: "G".into(),
            round_: "F".into(),
            best_of: 3,
            tourney_date: "2024-06-08".into(),
        };
        let payload = PredictPayload { matches: vec![req] };
        let json = serde_json::to_value(&payload).unwrap();
        assert!(json["matches"].is_array());
        assert_eq!(json["matches"].as_array().unwrap().len(), 1);
        assert_eq!(json["matches"][0]["tour"], "wta");
    }

    #[test]
    fn ml_match_prediction_deserializes_from_json() {
        let raw = r#"{
            "prob_a_wins": 0.65,
            "prob_a_wins_uncalibrated": 0.72,
            "elo_prob_a_wins": 0.60,
            "confidence_flag": "ok"
        }"#;
        let pred: MlMatchPrediction = serde_json::from_str(raw).unwrap();
        assert!((pred.prob_a_wins - 0.65).abs() < 1e-9);
        assert_eq!(pred.confidence_flag, "ok");
    }

    #[test]
    fn ml_predict_response_deserializes_from_json() {
        let raw = r#"{
            "predictions": [
                {
                    "prob_a_wins": 0.55,
                    "prob_a_wins_uncalibrated": 0.60,
                    "elo_prob_a_wins": 0.52,
                    "confidence_flag": "low_history"
                }
            ]
        }"#;
        let resp: MlPredictResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.predictions.len(), 1);
        assert_eq!(resp.predictions[0].confidence_flag, "low_history");
    }
}
