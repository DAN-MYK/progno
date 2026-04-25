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
use state::AppState;
#[cfg(not(test))]
use tauri::Manager;

fn main() {
    #[cfg(not(test))]
    tauri::Builder::default()
        .manage(AppState::default())
        .manage(std::sync::Mutex::new(sidecar::SidecarState::default()))
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            if let Ok(elo) = artifacts::load_elo_state_for_tour("atp") {
                *app.state::<AppState>().elo_atp.lock().expect("Mutex poisoned during setup") = Some(elo);
            }
            if let Ok(elo) = artifacts::load_elo_state_for_tour("wta") {
                *app.state::<AppState>().elo_wta.lock().expect("Mutex poisoned during setup") = Some(elo);
            }
            let artifacts_root = std::env::current_dir()
                .unwrap_or_default()
                .join("artifacts")
                .to_string_lossy()
                .to_string();
            sidecar::spawn_sidecar(&app.handle(), artifacts_root);
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
