#[cfg(test)]
mod tests {
    use progno::parser::parse_match_text;

    #[test]
    fn test_parse_simple_vs_format() {
        let text = "Alcaraz vs Sinner";
        let matches = parse_match_text(text).expect("should parse");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].player_a, "Alcaraz");
        assert_eq!(matches[0].player_b, "Sinner");
        assert_eq!(matches[0].surface, "Hard");  // default
    }

    #[test]
    fn test_parse_with_surface() {
        let text = "Alcaraz vs Sinner - Clay";
        let matches = parse_match_text(text).expect("should parse");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].surface.to_lowercase(), "clay");
    }

    #[test]
    fn test_parse_multiple_matches() {
        let text = "Alcaraz vs Sinner\nDjokovic vs Medvedev";
        let matches = parse_match_text(text).expect("should parse");
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].player_a, "Alcaraz");
        assert_eq!(matches[1].player_a, "Djokovic");
    }

    #[test]
    fn test_parse_hyphen_separator() {
        let text = "Alcaraz - Sinner";
        let matches = parse_match_text(text).expect("should parse");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].player_a, "Alcaraz");
    }

    #[test]
    fn test_parse_empty_returns_empty() {
        let text = "";
        let matches = parse_match_text(text).expect("should parse");
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_parse_v_abbreviation() {
        let text = "Federer v. Nadal - Grass";
        let matches = parse_match_text(text).expect("should parse");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].player_a, "Federer");
        assert_eq!(matches[0].player_b, "Nadal");
        assert_eq!(matches[0].surface, "Grass");
    }
}
