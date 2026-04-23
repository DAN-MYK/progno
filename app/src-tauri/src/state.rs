use std::sync::Mutex;

#[derive(Default)]
pub struct AppState {
    pub elo_state: Mutex<Option<serde_json::Value>>,
}
