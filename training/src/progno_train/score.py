"""Parse Sackmann score strings into structured data."""

from __future__ import annotations

import re
from dataclasses import dataclass

_RETIREMENT_MARKERS = frozenset({"RET", "W/O", "DEF", "DEF.", "ABN"})
SET_PATTERN = re.compile(r"^\[?(\d+)-(\d+)(?:\([^)]*\))?\]?$")


@dataclass(frozen=True)
class ParsedScore:
    is_complete: bool
    completed_sets: int
    winner_set_count: int
    loser_set_count: int


def parse_score(raw: str) -> ParsedScore:
    if not raw or not raw.strip():
        return ParsedScore(False, 0, 0, 0)

    tokens = raw.strip().split()
    tokens_upper = [t.upper() for t in tokens]

    is_retirement = any(t in _RETIREMENT_MARKERS for t in tokens_upper)
    set_matches = [m for tok in tokens if (m := SET_PATTERN.match(tok))]

    winner_sets = 0
    loser_sets = 0
    completed = 0
    for m in set_matches:
        winner_games, loser_games = int(m.group(1)), int(m.group(2))
        # Tied score (e.g. 6-6) means tiebreak in progress — not a completed set.
        if (winner_games >= 6 or loser_games >= 6) and winner_games != loser_games:
            completed += 1
            if winner_games > loser_games:
                winner_sets += 1
            else:
                loser_sets += 1

    if is_retirement:
        return ParsedScore(False, completed, winner_sets, loser_sets)

    if completed == 0:
        return ParsedScore(False, 0, 0, 0)

    return ParsedScore(True, completed, winner_sets, loser_sets)
