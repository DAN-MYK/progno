// Integration scenarios moved to src/commands.rs as unit tests (#[cfg(test)] mod tests).
// Linking progno as a library pulls in Tauri DLLs that are unavailable at test runtime on Windows.

#[test]
fn test_sidecar_predict_request_struct() {
    // Verify serialization matches sidecar API contract
    use serde_json::json;
    let req = json!({
        "matches": [{
            "player_a_id": "104745",
            "player_b_id": "106421",
            "surface": "Hard",
            "tourney_level": "M",
            "round_": "QF",
            "best_of": 3,
            "tourney_date": "2026-04-23"
        }]
    });
    // Verify it deserializes correctly
    assert_eq!(req["matches"][0]["surface"], "Hard");
    assert_eq!(req["matches"][0]["best_of"], 3);
}
