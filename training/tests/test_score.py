from __future__ import annotations

import pytest

from progno_train.score import ParsedScore, parse_score


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "6-4 3-6 7-5",
            ParsedScore(is_complete=True, completed_sets=3, winner_set_count=2, loser_set_count=1),
        ),
        (
            "6-4 6-3",
            ParsedScore(is_complete=True, completed_sets=2, winner_set_count=2, loser_set_count=0),
        ),
        (
            "7-6(4) 6-7(5) 6-4",
            ParsedScore(is_complete=True, completed_sets=3, winner_set_count=2, loser_set_count=1),
        ),
        (
            "6-4 3-6 RET",
            ParsedScore(is_complete=False, completed_sets=2, winner_set_count=1, loser_set_count=1),
        ),
        (
            "W/O",
            ParsedScore(is_complete=False, completed_sets=0, winner_set_count=0, loser_set_count=0),
        ),
        (
            "6-4 2-3 RET",
            ParsedScore(is_complete=False, completed_sets=1, winner_set_count=1, loser_set_count=0),
        ),
        (
            "6-3 3-6 6-2 4-6 8-6",
            ParsedScore(is_complete=True, completed_sets=5, winner_set_count=3, loser_set_count=2),
        ),
    ],
)
def test_parse_score(raw: str, expected: ParsedScore) -> None:
    assert parse_score(raw) == expected


def test_parse_score_none_or_empty() -> None:
    assert parse_score("").is_complete is False
    assert parse_score("").completed_sets == 0


def test_parse_score_def_treated_as_walkover() -> None:
    result = parse_score("DEF")
    assert result.is_complete is False
    assert result.completed_sets == 0


def test_parse_score_def_dot_mid_match() -> None:
    result = parse_score("6-4 Def.")
    assert result.is_complete is False
    assert result.completed_sets == 1


def test_parse_score_tied_set_not_counted_as_completed() -> None:
    result = parse_score("6-6 RET")
    assert result.is_complete is False
    assert result.completed_sets == 0
    assert result.winner_set_count == 0
    assert result.loser_set_count == 0


def test_parse_score_match_tiebreak_bracket_format() -> None:
    result = parse_score("6-3 4-6 [10-7]")
    assert result == ParsedScore(
        is_complete=True, completed_sets=3, winner_set_count=2, loser_set_count=1
    )


def test_parse_score_abn_mid_match() -> None:
    result = parse_score("6-4 2-3 ABN")
    assert result.is_complete is False
    assert result.completed_sets == 1
    assert result.winner_set_count == 1
    assert result.loser_set_count == 0
