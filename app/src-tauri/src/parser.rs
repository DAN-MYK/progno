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
        _ => "Hard".to_string(),
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

    let vs_pattern = Regex::new(r"(.+?)\s+(?:vs|v\.?|-)\s+(.+?)(?:\s*-\s*(.+?))?$")
        .map_err(|e| format!("Regex error: {}", e))?;

    for line in lines {
        let line = line.trim();

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_vs_format() {
        let matches = parse_match_text("Alcaraz vs Sinner").expect("should parse");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].player_a, "Alcaraz");
        assert_eq!(matches[0].player_b, "Sinner");
        assert_eq!(matches[0].surface, "Hard");
    }

    #[test]
    fn test_parse_with_surface() {
        let matches = parse_match_text("Alcaraz vs Sinner - Clay").expect("should parse");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].surface.to_lowercase(), "clay");
    }

    #[test]
    fn test_parse_multiple_matches() {
        let matches = parse_match_text("Alcaraz vs Sinner\nDjokovic vs Medvedev").expect("should parse");
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].player_a, "Alcaraz");
        assert_eq!(matches[1].player_a, "Djokovic");
    }

    #[test]
    fn test_parse_hyphen_separator() {
        let matches = parse_match_text("Alcaraz - Sinner").expect("should parse");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].player_a, "Alcaraz");
    }

    #[test]
    fn test_parse_empty_returns_empty() {
        let matches = parse_match_text("").expect("should parse");
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_parse_v_abbreviation() {
        let matches = parse_match_text("Federer v. Nadal - Grass").expect("should parse");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].player_a, "Federer");
        assert_eq!(matches[0].player_b, "Nadal");
        assert_eq!(matches[0].surface, "Grass");
    }
}
