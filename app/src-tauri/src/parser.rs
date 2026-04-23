pub struct ParsedMatch {
    pub player_a: String,
    pub player_b: String,
    pub surface: String,
}

pub fn parse_match_text(text: &str) -> Result<Vec<ParsedMatch>, String> {
    Ok(vec![])
}
