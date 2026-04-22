"""Elo rating updates (FiveThirtyEight tennis variant).

K-factor formula: K(n) = 250 / (n + 5)^0.4  where n = player's prior match count.

Reference: FiveThirtyEight tennis Elo methodology. The formula is multiplied by
context factors (tournament level, round, best_of) at callsites in `rollup.py`.
"""

from __future__ import annotations


INITIAL_RATING = 1500.0
K_BASE = 250.0
K_OFFSET = 5.0
K_SHAPE = 0.4


def expected_probability(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def k_factor(prior_matches: int) -> float:
    if prior_matches < 0:
        raise ValueError(f"prior_matches must be >= 0, got {prior_matches}")
    return K_BASE / (prior_matches + K_OFFSET) ** K_SHAPE


def apply_elo_update(
    winner_rating: float,
    loser_rating: float,
    k: float,
) -> tuple[float, float]:
    expected_w = expected_probability(winner_rating, loser_rating)
    delta = k * (1.0 - expected_w)
    return winner_rating + delta, loser_rating - delta
