from __future__ import annotations

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
    new_winner, new_loser = apply_elo_update(winner_rating=1500, loser_rating=1500, k=32)
    assert new_winner == pytest.approx(1516.0)
    assert new_loser == pytest.approx(1484.0)


def test_apply_elo_update_upset_winner_gains_more() -> None:
    # Lower-rated player wins → gains more than they would at parity
    new_winner, new_loser = apply_elo_update(winner_rating=1400, loser_rating=1700, k=32)
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


from progno_train.elo import context_multiplier  # noqa: E402


def test_context_multiplier_grand_slam_bo5_final() -> None:
    # G (1.0) * F (1.0) * BO5 (1.0) = 1.0
    assert context_multiplier("G", "F", 5) == pytest.approx(1.0)


def test_context_multiplier_masters_early_round_bo3() -> None:
    # M (0.85) * R32 (0.85) * BO3 (0.90) = 0.65025
    assert context_multiplier("M", "R32", 3) == pytest.approx(0.85 * 0.85 * 0.90)


def test_context_multiplier_challenger_qualifier_bo3() -> None:
    # C (0.50) * Q (0.85) * BO3 (0.90)
    assert context_multiplier("C", "Q1", 3) == pytest.approx(0.50 * 0.85 * 0.90)


def test_context_multiplier_unknown_level_falls_back_to_atp() -> None:
    # Unknown level defaults to A (0.75)
    assert context_multiplier("UNKNOWN", "R16", 3) == pytest.approx(0.75 * 0.85 * 0.90)


@pytest.mark.parametrize(
    ("level", "round_", "best_of"),
    [
        ("G", "F", 5),
        ("M", "SF", 3),
        ("A", "QF", 3),
    ],
)
def test_context_multiplier_positive(level: str, round_: str, best_of: int) -> None:
    assert context_multiplier(level, round_, best_of) > 0.0


from progno_train.elo import apply_welo_update, mov_multiplier  # noqa: E402


def test_welo_mov_multiplier_values() -> None:
    # 3-0 or 2-0 → ratio=1.0 → 2*1.0-0.5=1.50
    assert mov_multiplier(3, 0) == pytest.approx(1.50)
    assert mov_multiplier(2, 0) == pytest.approx(1.50)
    # 3-1 → ratio=0.75 → 2*0.75-0.5=1.00
    assert mov_multiplier(3, 1) == pytest.approx(1.00)
    # 3-2 → ratio=3/5=0.60 → 2*0.60-0.5=0.70
    assert mov_multiplier(3, 2) == pytest.approx(0.70)
    # 2-1 → ratio=2/3 → 2*(2/3)-0.5=5/6≈0.8333
    assert mov_multiplier(2, 1) == pytest.approx(2.0 * (2 / 3) - 0.5)
    # degenerate: both 0 → 1.0 (neutral fallback)
    assert mov_multiplier(0, 0) == pytest.approx(1.0)


def test_welo_dominant_win_higher_k() -> None:
    # 3-0 win (multiplier=1.5) → winner gains more than standard Elo (multiplier=1.0)
    std_w, _ = apply_elo_update(1500, 1500, k=32)
    welo_w, _ = apply_welo_update(1500, 1500, k=32, sets_w=3, sets_l=0)
    assert welo_w > std_w


def test_welo_close_win_lower_k() -> None:
    # 3-2 win (multiplier=0.7) → winner gains less than standard Elo (multiplier=1.0)
    std_w, _ = apply_elo_update(1500, 1500, k=32)
    welo_w, _ = apply_welo_update(1500, 1500, k=32, sets_w=3, sets_l=2)
    assert welo_w < std_w


def test_welo_total_mass_not_conserved() -> None:
    # WElo is non-constant: dominant wins move more rating than close wins
    nw_30, nl_30 = apply_welo_update(1500, 1500, k=32, sets_w=3, sets_l=0)
    nw_32, nl_32 = apply_welo_update(1500, 1500, k=32, sets_w=3, sets_l=2)
    assert nw_30 > 1500          # winner always gains
    assert nw_32 > 1500          # winner gains even in close match
    assert nw_30 - 1500 > nw_32 - 1500  # dominant win → larger rating jump
