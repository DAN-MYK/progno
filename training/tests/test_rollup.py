from __future__ import annotations

import pandas as pd
import pytest

from progno_train.elo import INITIAL_RATING
from progno_train.rollup import rollup_elo


def _mk_match(
    winner: int,
    loser: int,
    date: str = "2020-01-01",
    surface: str = "Hard",
    level: str = "A",
    round_: str = "R32",
    best_of: int = 3,
    is_complete: bool = True,
    match_num: int = 1,
) -> dict:
    return {
        "tourney_id": f"{date}-T",
        "tourney_date": pd.Timestamp(date),
        "match_num": match_num,
        "surface": surface,
        "tourney_level": level,
        "round": round_,
        "best_of": best_of,
        "winner_id": winner,
        "loser_id": loser,
        "is_complete": is_complete,
    }


def test_rollup_empty_returns_empty_state() -> None:
    df = pd.DataFrame(columns=["tourney_date", "match_num"])
    state = rollup_elo(df)
    assert state == {}


def test_rollup_single_match_updates_both_players_overall_and_surface() -> None:
    df = pd.DataFrame([_mk_match(1, 2, surface="Clay")])
    state = rollup_elo(df)
    assert 1 in state
    assert 2 in state
    assert state[1].elo_overall > INITIAL_RATING
    assert state[2].elo_overall < INITIAL_RATING
    assert state[1].elo_clay > INITIAL_RATING
    assert state[1].elo_hard == INITIAL_RATING
    assert state[1].elo_grass == INITIAL_RATING
    assert state[1].matches_played == 1
    assert state[2].matches_played == 1


def test_rollup_carpet_updates_only_overall() -> None:
    df = pd.DataFrame([_mk_match(1, 2, surface="Carpet")])
    state = rollup_elo(df)
    assert state[1].elo_overall > INITIAL_RATING
    assert state[1].elo_hard == INITIAL_RATING
    assert state[1].elo_clay == INITIAL_RATING
    assert state[1].elo_grass == INITIAL_RATING


def test_rollup_skips_incomplete_matches() -> None:
    df = pd.DataFrame(
        [
            _mk_match(1, 2, is_complete=False, match_num=1),
            _mk_match(3, 4, is_complete=True, match_num=2),
        ]
    )
    state = rollup_elo(df)
    assert 1 not in state
    assert 2 not in state
    assert 3 in state
    assert 4 in state


def test_rollup_unknown_surface_updates_only_overall() -> None:
    df = pd.DataFrame([_mk_match(1, 2, surface="")])
    state = rollup_elo(df)
    assert state[1].elo_overall > INITIAL_RATING
    assert state[1].elo_hard == INITIAL_RATING


def test_rollup_order_matters_matches_sorted_by_date_then_match_num() -> None:
    # If rollup is processing out of order we'd see different ratings
    df = pd.DataFrame(
        [
            _mk_match(1, 2, date="2020-01-01", match_num=1),
            _mk_match(1, 2, date="2020-01-01", match_num=2),
            _mk_match(1, 2, date="2020-01-01", match_num=3),
        ]
    )
    state = rollup_elo(df)
    assert state[1].matches_played == 3
    assert state[2].matches_played == 3


def test_rollup_deterministic() -> None:
    df = pd.DataFrame(
        [
            _mk_match(1, 2, match_num=1),
            _mk_match(3, 4, match_num=2),
            _mk_match(1, 3, match_num=3),
        ]
    )
    a = rollup_elo(df)
    b = rollup_elo(df)
    for pid in a:
        assert a[pid].elo_overall == pytest.approx(b[pid].elo_overall)
        assert a[pid].matches_played == b[pid].matches_played
