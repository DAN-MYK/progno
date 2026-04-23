#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod artifacts;
mod commands;
mod elo;
mod parser;

use tauri::{Manager, State};

#[derive(Default)]
struct AppState {
    elo_state: Option<serde_json::Value>,
}

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::parse_and_predict,
            commands::get_data_as_of,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
