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
