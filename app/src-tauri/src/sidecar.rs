use std::sync::Mutex;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use reqwest::Client;

pub struct SidecarState {
    pub port: Option<u16>,
    _child: Option<CommandChild>,
    pub client: Client,
}

impl Default for SidecarState {
    fn default() -> Self {
        Self {
            port: None,
            _child: None,
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
        }
    }
}

#[derive(Serialize, Clone)]
pub struct MlMatchRequest {
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

#[derive(Deserialize, Clone)]
pub struct MlMatchPrediction {
    pub prob_a_wins: f64,
    pub prob_a_wins_uncalibrated: f64,
    pub elo_prob_a_wins: f64,
    pub confidence_flag: String,
}

#[derive(Deserialize)]
pub struct MlPredictResponse {
    pub model_version: String,
    pub predictions: Vec<MlMatchPrediction>,
}

pub fn spawn_sidecar(app: &tauri::AppHandle, artifacts_dir: String) {
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        match do_spawn(&handle, &artifacts_dir).await {
            Ok((port, child)) => {
                let state = handle.state::<Mutex<SidecarState>>();
                if let Ok(mut s) = state.lock() {
                    s.port = Some(port);
                    s._child = Some(child);
                }
                eprintln!("[sidecar] ready on port {port}");
            }
            Err(e) => eprintln!("[sidecar] failed to start: {e}"),
        }
    });
}

async fn do_spawn(app: &tauri::AppHandle, artifacts_dir: &str) -> Result<(u16, CommandChild)> {
    let (mut rx, child) = app
        .shell()
        .sidecar("progno-sidecar")?
        .args(["--artifacts-dir", artifacts_dir])
        .spawn()?;

    while let Some(event) = rx.recv().await {
        match event {
            CommandEvent::Stdout(line) => {
                let s = String::from_utf8_lossy(&line);
                if let Some(port_str) = s.trim().strip_prefix("READY port=") {
                    let port: u16 = port_str.parse()?;
                    return Ok((port, child));
                }
            }
            CommandEvent::Terminated(status) => {
                return Err(anyhow::anyhow!("sidecar exited: {:?}", status));
            }
            _ => {}
        }
    }
    Err(anyhow::anyhow!("sidecar did not emit READY"))
}

pub async fn ml_predict(client: &Client, port: u16, matches: Vec<MlMatchRequest>) -> Result<MlPredictResponse> {
    let url = format!("http://127.0.0.1:{port}/predict");
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
