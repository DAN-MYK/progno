#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod artifacts;
mod commands;
mod elo;
mod parser;
mod state;

#[cfg(not(test))]
use state::AppState;
#[cfg(not(test))]
use tauri::Manager;

fn main() {
    #[cfg(not(test))]
    tauri::Builder::default()
        .manage(AppState::default())
        .setup(|app| {
            let path = artifacts::elo_state_path();
            if let Ok(elo) = artifacts::load_elo_state(path.to_str().unwrap_or("elo_state.json")) {
                *app.state::<AppState>().elo_state.lock().unwrap() = Some(elo);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::parse_and_predict,
            commands::get_data_as_of_cmd,
        ])
        .run(tauri::generate_context!())
        .map_err(|e| eprintln!("Failed to run Tauri: {}", e))
        .ok();
}
