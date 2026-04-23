/// Calculate expected win probability using Elo formula.
/// P(A wins) = 1 / (1 + 10^((elo_B - elo_A) / 400))
pub fn expected_probability(rating_a: f64, rating_b: f64) -> f64 {
    1.0 / (1.0 + 10.0_f64.powf((rating_b - rating_a) / 400.0))
}
