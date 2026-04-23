#[cfg(test)]
mod tests {
    use progno::elo::{expected_probability, surface_elo};

    #[test]
    fn test_expected_prob_equal_ratings() {
        let p = expected_probability(1500.0, 1500.0);
        assert!((p - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_expected_prob_higher_favored() {
        let p = expected_probability(1700.0, 1500.0);
        assert!(p > 0.74 && p < 0.77);
    }

    #[test]
    fn test_expected_prob_symmetric() {
        let p_ab = expected_probability(1600.0, 1500.0);
        let p_ba = expected_probability(1500.0, 1600.0);
        assert!((p_ab + p_ba - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_surface_elo_pure_overall_low_history() {
        // <20 matches on surface → pure overall
        let elo = surface_elo(1500.0, 1600.0, 5);
        assert_eq!(elo, 1600.0);
    }

    #[test]
    fn test_surface_elo_composite_high_history() {
        // ≥20 matches → 0.5 * surface + 0.5 * overall
        let elo = surface_elo(1600.0, 1500.0, 25);
        assert_eq!(elo, 1550.0);
    }

    #[test]
    fn test_surface_elo_boundary_19_vs_20() {
        let elo_19 = surface_elo(1600.0, 1500.0, 19);
        let elo_20 = surface_elo(1600.0, 1500.0, 20);
        assert_eq!(elo_19, 1500.0);
        assert!((elo_20 - 1550.0).abs() < 0.001);
    }
}
