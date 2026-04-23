#[tauri::command]
pub fn parse_and_predict(text: String) -> String {
    format!("Parsed: {}", text)
}

#[tauri::command]
pub fn get_data_as_of() -> String {
    "2026-04-20".to_string()
}
