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


LEVEL_FACTORS = {
    "G": 1.00,  # Grand Slam
    "M": 0.85,  # Masters 1000
    "A": 0.75,  # ATP 250 / 500 (default)
    "F": 0.90,  # Tour Finals
    "D": 0.70,  # Davis Cup
    "C": 0.50,  # Challenger
    "S": 0.40,  # ITF Satellite
}

ROUND_FACTORS = {
    "F": 1.00,
    "SF": 0.95,
    "QF": 0.90,
    "R16": 0.85,
    "R32": 0.85,
    "R64": 0.85,
    "R128": 0.85,
    "RR": 0.90,  # round robin
    "BR": 0.85,  # bronze
    "ER": 0.85,  # early round
}


def _round_factor(round_: str) -> float:
    r = (round_ or "").strip().upper()
    if r in ROUND_FACTORS:
        return ROUND_FACTORS[r]
    if r.startswith("Q"):
        return 0.85
    return 0.85


def context_multiplier(tourney_level: str, round_: str, best_of: int) -> float:
    lf = LEVEL_FACTORS.get((tourney_level or "").strip().upper(), LEVEL_FACTORS["A"])
    rf = _round_factor(round_)
    bo5 = 1.0 if best_of == 5 else 0.90
    return lf * rf * bo5


def mov_multiplier(sets_winner: int, sets_loser: int) -> float:
    """Map set ratio [0.5, 1.0] to K multiplier [0.50, 1.50].

    Returns 1.0 for degenerate inputs (no sets played).
    Examples: 3-0→1.50, 3-1→1.00, 3-2→0.70, 2-0→1.50, 2-1≈0.83.
    """
    total = sets_winner + sets_loser
    if total <= 0:
        return 1.0
    return 2.0 * (sets_winner / total) - 0.5


def apply_welo_update(
    winner_rating: float,
    loser_rating: float,
    k: float,
    sets_w: int,
    sets_l: int,
) -> tuple[float, float]:
    """WElo: apply_elo_update with K scaled by margin-of-victory multiplier."""
    return apply_elo_update(winner_rating, loser_rating, k=k * mov_multiplier(sets_w, sets_l))
