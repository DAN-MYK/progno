"""Parse Sackmann score strings into structured data."""

from __future__ import annotations

import re
from dataclasses import dataclass

RETIREMENT_MARKERS = ("RET", "W/O", "DEF", "Def.", "ABN")
SET_PATTERN = re.compile(r"^(\d+)-(\d+)(?:\([^)]*\))?$")


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

    is_retirement = any(t in RETIREMENT_MARKERS for t in tokens_upper)
    set_tokens = [t for t in tokens if SET_PATTERN.match(t)]

    winner_sets = 0
    loser_sets = 0
    completed = 0
    for tok in set_tokens:
        m = SET_PATTERN.match(tok)
        assert m is not None
        winner_games, loser_games = int(m.group(1)), int(m.group(2))
        # Set is complete if one player has 6+ games (covers 6+ with 2-game lead or 7+ tiebreak).
        if winner_games >= 6 or loser_games >= 6:
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
