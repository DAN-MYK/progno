#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod artifacts;
mod commands;
mod elo;
mod kelly;
mod parser;
mod sidecar;
mod state;

#[cfg(not(test))]
use std::sync::Mutex;
#[cfg(not(test))]
use state::AppState;
#[cfg(not(test))]
use tauri::Manager;

fn main() {
    #[cfg(not(test))]
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(AppState::default())
        .manage(Mutex::new(sidecar::SidecarState::default()))
        .setup(|app| {
            let path = artifacts::elo_state_path();
            if let Ok(elo) = artifacts::load_elo_state(path.to_str().unwrap_or("elo_state.json")) {
                *app.state::<AppState>().elo_state.lock().unwrap() = Some(elo);
            }
            let artifacts_dir = path
                .parent()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|| ".".to_string());
            sidecar::spawn_sidecar(&app.handle(), artifacts_dir);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::parse_and_predict,
            commands::get_data_as_of_cmd,
            commands::calculate_kelly,
            commands::predict_with_ml,
        ])
        .run(tauri::generate_context!())
        .map_err(|e| eprintln!("Failed to run Tauri: {}", e))
        .ok();
}
