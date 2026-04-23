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
