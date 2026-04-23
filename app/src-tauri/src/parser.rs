use regex::Regex;

#[derive(Debug, Clone)]
pub struct ParsedMatch {
    pub player_a: String,
    pub player_b: String,
    pub surface: String,
}

fn normalize_surface(s: &str) -> String {
    match s.to_lowercase().trim() {
        "hard" => "Hard".to_string(),
        "clay" => "Clay".to_string(),
        "grass" => "Grass".to_string(),
        other => {
            if other.is_empty() {
                "Hard".to_string()
            } else {
                other.to_string()
            }
        }
    }
}

/// Parse match text and extract player names and surface.
pub fn parse_match_text(text: &str) -> Result<Vec<ParsedMatch>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(vec![]);
    }

    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut matches = Vec::new();

    for line in lines {
        let line = line.trim();

        let vs_pattern = Regex::new(r"(.+?)\s+(?:vs|v\.?|-)\s+(.+?)(?:\s*-\s*(.+?))?$")
            .map_err(|e| format!("Regex error: {}", e))?;

        if let Some(caps) = vs_pattern.captures(line) {
            let player_a = caps.get(1).map(|m| m.as_str()).unwrap_or("").trim().to_string();
            let player_b = caps.get(2).map(|m| m.as_str()).unwrap_or("").trim().to_string();
            let surface_text = caps.get(3).map(|m| m.as_str()).unwrap_or("").trim();
            let surface = normalize_surface(surface_text);

            if !player_a.is_empty() && !player_b.is_empty() {
                matches.push(ParsedMatch {
                    player_a,
                    player_b,
                    surface,
                });
            }
        }
    }

    Ok(matches)
}
