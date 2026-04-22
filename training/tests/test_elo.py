from __future__ import annotations

import math

import pytest

from progno_train.elo import (
    apply_elo_update,
    expected_probability,
    k_factor,
)


def test_expected_prob_equal_ratings() -> None:
    assert expected_probability(1500, 1500) == pytest.approx(0.5)


def test_expected_prob_higher_rating_favoured() -> None:
    p = expected_probability(1700, 1500)
    assert 0.74 < p < 0.77  # classical 200 gap ≈ 0.76


def test_expected_prob_symmetric() -> None:
    p_ab = expected_probability(1600, 1500)
    p_ba = expected_probability(1500, 1600)
    assert p_ab + p_ba == pytest.approx(1.0)


def test_k_factor_formula_538() -> None:
    # K(n) = 250 / (n + 5)^0.4
    assert k_factor(0) == pytest.approx(250 / 5**0.4)
    assert k_factor(100) == pytest.approx(250 / 105**0.4)


def test_k_factor_decreases_with_experience() -> None:
    assert k_factor(0) > k_factor(10) > k_factor(100) > k_factor(1000)


def test_apply_elo_update_equal_ratings_winner_gains_k_half() -> None:
    new_winner, new_loser = apply_elo_update(
        winner_rating=1500, loser_rating=1500, k=32
    )
    assert new_winner == pytest.approx(1516.0)
    assert new_loser == pytest.approx(1484.0)


def test_apply_elo_update_upset_winner_gains_more() -> None:
    # Lower-rated player wins → gains more than they would at parity
    new_winner, new_loser = apply_elo_update(
        winner_rating=1400, loser_rating=1700, k=32
    )
    gain = new_winner - 1400
    assert gain > 16.0


def test_apply_elo_update_preserves_total() -> None:
    # Elo is zero-sum when K is identical
    for wr, lr in [(1500, 1500), (1800, 1400), (1200, 1700)]:
        nw, nl = apply_elo_update(wr, lr, k=32)
        assert (nw + nl) == pytest.approx(wr + lr)


def test_apply_elo_update_k_zero_no_change() -> None:
    nw, nl = apply_elo_update(1500, 1600, k=0)
    assert nw == 1500
    assert nl == 1600


def test_apply_elo_update_returns_floats() -> None:
    nw, nl = apply_elo_update(1500, 1500, k=32)
    assert isinstance(nw, float)
    assert isinstance(nl, float)


def test_k_factor_rejects_negative_n() -> None:
    with pytest.raises(ValueError):
        k_factor(-1)
