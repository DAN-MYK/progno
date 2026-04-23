#[cfg(test)]
mod tests {
    use progno::kelly::*;

    #[test]
    fn test_implied_probability_from_decimal_odds() {
        // 2.0 decimal odds → 50% implied
        assert!((implied_probability(2.0) - 0.5).abs() < 0.001);
        // 1.5 decimal odds → 66.67% implied
        assert!((implied_probability(1.5) - (2.0 / 3.0)).abs() < 0.001);
        // 3.0 decimal odds → 33.33% implied
        assert!((implied_probability(3.0) - (1.0 / 3.0)).abs() < 0.001);
    }

    #[test]
    fn test_edge_calculation() {
        // If model says 60% but implied is 50%, edge is +10%
        let edge_val = edge(0.6, 2.0);
        assert!((edge_val - 0.1).abs() < 0.001);

        // If model says 30% but implied is 50%, edge is -20%
        let edge_val2 = edge(0.3, 2.0);
        assert!((edge_val2 - (-0.2)).abs() < 0.001);
    }

    #[test]
    fn test_full_kelly_fraction_positive_edge() {
        // Model 60%, decimal_odds 2.0
        // full_kelly = (0.6 * 2.0 - 1) / (2.0 - 1) = 0.2 / 1.0 = 0.2
        let kelly = full_kelly_fraction(0.6, 2.0);
        assert!((kelly - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_full_kelly_fraction_negative_edge() {
        // Model 40%, decimal_odds 2.0
        // full_kelly = (0.4 * 2.0 - 1) / (2.0 - 1) = -0.2 / 1.0 = -0.2
        let kelly = full_kelly_fraction(0.4, 2.0);
        assert!((kelly - (-0.2)).abs() < 0.001);
    }

    #[test]
    fn test_fractional_kelly_applies_fraction() {
        // Model 60%, decimal_odds 2.0, fraction 0.25
        // full = 0.2, frac = 0.25 * 0.2 = 0.05
        let kelly = fractional_kelly(0.6, 2.0, 0.25);
        assert!((kelly - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_fractional_kelly_clamps_negative_to_zero() {
        // Model 30%, decimal_odds 2.0, fraction 0.25
        // full = -0.5, frac = max(0, 0.25 * -0.5) = 0.0
        let kelly = fractional_kelly(0.3, 2.0, 0.25);
        assert_eq!(kelly, 0.0);
    }

    #[test]
    fn test_stake_from_kelly() {
        // bankroll $1000, kelly 0.05 → $50
        let stake = stake_from_kelly(1000.0, 0.05);
        assert!((stake - 50.0).abs() < 0.01);
    }
}
