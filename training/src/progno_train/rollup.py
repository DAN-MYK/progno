"""Roll up matches into per-player Elo state (overall + surface-specific)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from progno_train.elo import (
    INITIAL_RATING,
    apply_elo_update,
    apply_welo_update,
    context_multiplier,
    k_factor,
)

TRACKED_SURFACES = {"Hard", "Clay", "Grass"}


@dataclass
class PlayerElo:
    player_id: int
    elo_overall: float = INITIAL_RATING
    elo_hard: float = INITIAL_RATING
    elo_clay: float = INITIAL_RATING
    elo_grass: float = INITIAL_RATING
    welo_overall: float = INITIAL_RATING
    welo_hard: float = INITIAL_RATING
    welo_clay: float = INITIAL_RATING
    welo_grass: float = INITIAL_RATING
    matches_played: int = 0
    matches_played_hard: int = 0
    matches_played_clay: int = 0
    matches_played_grass: int = 0


def _get_or_init(state: dict[int, PlayerElo], pid: int) -> PlayerElo:
    if pid not in state:
        state[pid] = PlayerElo(player_id=pid)
    return state[pid]


def _update_surface(
    winner: PlayerElo,
    loser: PlayerElo,
    surface: str,
    k_with_context: float,
) -> None:
    if surface not in TRACKED_SURFACES:
        return
    attr = f"elo_{surface.lower()}"
    played_attr = f"matches_played_{surface.lower()}"
    new_w, new_l = apply_elo_update(
        getattr(winner, attr),
        getattr(loser, attr),
        k=k_with_context,
    )
    setattr(winner, attr, new_w)
    setattr(loser, attr, new_l)
    setattr(winner, played_attr, getattr(winner, played_attr) + 1)
    setattr(loser, played_attr, getattr(loser, played_attr) + 1)


def _update_welo_surface(
    winner: PlayerElo,
    loser: PlayerElo,
    surface: str,
    k: float,
    sets_w: int,
    sets_l: int,
) -> None:
    if surface not in TRACKED_SURFACES:
        return
    attr = f"welo_{surface.lower()}"
    new_w, new_l = apply_welo_update(
        getattr(winner, attr),
        getattr(loser, attr),
        k=k,
        sets_w=sets_w,
        sets_l=sets_l,
    )
    setattr(winner, attr, new_w)
    setattr(loser, attr, new_l)


def rollup_elo(matches: pd.DataFrame) -> dict[int, PlayerElo]:
    if matches.empty:
        return {}

    required = {
        "tourney_date",
        "match_num",
        "winner_id",
        "loser_id",
        "surface",
        "tourney_level",
        "round",
        "best_of",
        "is_complete",
    }
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches DataFrame missing columns: {missing}")

    df = matches.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    state: dict[int, PlayerElo] = {}

    for row in df.itertuples(index=False):
        if not row.is_complete:
            continue

        winner = _get_or_init(state, int(row.winner_id))
        loser = _get_or_init(state, int(row.loser_id))

        ctx = context_multiplier(row.tourney_level, row.round, int(row.best_of))
        # K uses overall match count (538 convention); min of winner/loser is conservative
        k = min(k_factor(winner.matches_played), k_factor(loser.matches_played)) * ctx

        new_w, new_l = apply_elo_update(winner.elo_overall, loser.elo_overall, k=k)
        winner.elo_overall = new_w
        loser.elo_overall = new_l
        winner.matches_played += 1
        loser.matches_played += 1

        _update_surface(winner, loser, row.surface, k_with_context=k)

        _raw_w = getattr(row, "w_sets", None)
        sets_w = 0 if _raw_w is None or pd.isna(_raw_w) else int(_raw_w)
        _raw_l = getattr(row, "l_sets", None)
        sets_l = 0 if _raw_l is None or pd.isna(_raw_l) else int(_raw_l)
        new_ww, new_wl = apply_welo_update(
            winner.welo_overall, loser.welo_overall, k=k, sets_w=sets_w, sets_l=sets_l
        )
        winner.welo_overall = new_ww
        loser.welo_overall = new_wl
        _update_welo_surface(winner, loser, row.surface, k=k, sets_w=sets_w, sets_l=sets_l)

    return state
