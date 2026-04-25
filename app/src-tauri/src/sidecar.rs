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
