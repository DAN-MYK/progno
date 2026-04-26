/// Implied probability from decimal odds.
/// odds = 2.0 → 0.5 (50%)
pub fn implied_probability(decimal_odds: f64) -> f64 {
    1.0 / decimal_odds
}

/// Expected value edge: model probability minus implied probability.
pub fn edge(model_prob: f64, decimal_odds: f64) -> f64 {
    model_prob - implied_probability(decimal_odds)
}

/// Full Kelly fraction: (p * odds - 1) / (odds - 1)
pub fn full_kelly_fraction(model_prob: f64, decimal_odds: f64) -> f64 {
    (model_prob * decimal_odds - 1.0) / (decimal_odds - 1.0)
}

/// Fractional Kelly: max(0, fraction * full_kelly)
pub fn fractional_kelly(model_prob: f64, decimal_odds: f64, fraction: f64) -> f64 {
    (fraction * full_kelly_fraction(model_prob, decimal_odds)).max(0.0)
}

/// Stake in currency units from bankroll and Kelly fraction.
pub fn stake_from_kelly(bankroll: f64, kelly_fraction: f64) -> f64 {
    bankroll * kelly_fraction
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn implied_probability_even_odds() {
        assert!(approx(implied_probability(2.0), 0.5));
    }

    #[test]
    fn implied_probability_four_to_one() {
        assert!(approx(implied_probability(4.0), 0.25));
    }

    #[test]
    fn edge_positive_when_model_beats_implied() {
        assert!(approx(edge(0.6, 2.0), 0.1));
    }

    #[test]
    fn edge_negative_when_model_below_implied() {
        assert!(approx(edge(0.4, 2.0), -0.1));
    }

    #[test]
    fn edge_zero_when_model_equals_implied() {
        assert!(approx(edge(0.5, 2.0), 0.0));
    }

    #[test]
    fn full_kelly_zero_edge_gives_zero() {
        // model=0.5, odds=2.0 → (0.5*2-1)/(2-1) = 0
        assert!(approx(full_kelly_fraction(0.5, 2.0), 0.0));
    }

    #[test]
    fn full_kelly_positive_edge() {
        // model=0.6, odds=2.0 → (0.6*2-1)/(2-1) = 0.2
        assert!(approx(full_kelly_fraction(0.6, 2.0), 0.2));
    }

    #[test]
    fn fractional_kelly_clamps_negative_to_zero() {
        // negative edge → full_kelly negative → fractional = 0
        assert_eq!(fractional_kelly(0.4, 2.0, 0.25), 0.0);
    }

    #[test]
    fn fractional_kelly_quarter_of_full() {
        // full_kelly=0.2, fraction=0.25 → 0.05
        assert!(approx(fractional_kelly(0.6, 2.0, 0.25), 0.05));
    }

    #[test]
    fn stake_from_kelly_scales_bankroll() {
        assert!(approx(stake_from_kelly(1000.0, 0.05), 50.0));
    }

    #[test]
    fn stake_from_kelly_zero_fraction() {
        assert_eq!(stake_from_kelly(1000.0, 0.0), 0.0);
    }
}
