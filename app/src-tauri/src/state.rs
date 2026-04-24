use std::sync::Mutex;

#[derive(Default)]
pub struct AppState {
    pub elo_atp: Mutex<Option<serde_json::Value>>,
    pub elo_wta: Mutex<Option<serde_json::Value>>,
}
