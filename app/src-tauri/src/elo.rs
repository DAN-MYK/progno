/// Calculate expected win probability using Elo formula.
/// P(A wins) = 1 / (1 + 10^((elo_B - elo_A) / 400))
pub fn expected_probability(rating_a: f64, rating_b: f64) -> f64 {
    1.0 / (1.0 + 10.0_f64.powf((rating_b - rating_a) / 400.0))
}

/// Get composite Elo for a surface, applying weighting if enough history.
/// If matches_on_surface >= 20: composite = 0.5 * surface + 0.5 * overall
/// Otherwise: composite = overall
pub fn surface_elo(elo_surface: f64, elo_overall: f64, matches_on_surface: u32) -> f64 {
    if matches_on_surface >= 20 {
        0.5 * elo_surface + 0.5 * elo_overall
    } else {
        elo_overall
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let elo = surface_elo(1500.0, 1600.0, 5);
        assert_eq!(elo, 1600.0);
    }

    #[test]
    fn test_surface_elo_composite_high_history() {
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
