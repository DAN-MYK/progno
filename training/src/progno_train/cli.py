"""CLI entry points for the training pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from progno_train.artifacts import write_elo_state, write_match_history, write_players
from progno_train.config import Paths
from progno_train.ingest import ingest_sackmann_csv
from progno_train.rollup import rollup_elo

log = logging.getLogger("progno_train")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def run_update_data(paths: Paths) -> int:
    log.info("update_data is a stub in Phase 1a; expected to be implemented later")
    log.info("expected input: Sackmann CSVs placed at %s", paths.data_raw)
    if not paths.data_raw.exists():
        paths.data_raw.mkdir(parents=True, exist_ok=True)
    return 0


def run_ingest(paths: Paths) -> int:
    csvs = sorted(paths.data_raw.glob("atp_matches_*.csv"))
    if not csvs:
        log.error("no Sackmann CSVs found in %s", paths.data_raw)
        return 2
    log.info("ingesting %d CSV files", len(csvs))
    df = ingest_sackmann_csv(csvs)
    out = paths.data_staging / "matches_clean.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("wrote %d rows to %s", len(df), out)
    return 0


def run_elo(paths: Paths) -> int:
    staging = paths.data_staging / "matches_clean.parquet"
    if not staging.exists():
        log.error("no staging parquet at %s; run ingest first", staging)
        return 2
    matches = pd.read_parquet(staging)
    log.info("rolling up %d matches", len(matches))
    state = rollup_elo(matches)
    log.info("produced Elo state for %d players", len(state))

    data_as_of = matches["tourney_date"].max()
    all_names = pd.concat([
        matches[["winner_id", "winner_name"]].rename(columns={"winner_id": "id", "winner_name": "name"}),
        matches[["loser_id", "loser_name"]].rename(columns={"loser_id": "id", "loser_name": "name"}),
    ]).drop_duplicates("id")
    player_names = {int(row.id): row.name.split()[-1].lower() for row in all_names.itertuples()}
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    write_elo_state(state, paths.artifacts / "elo_state.json", data_as_of=data_as_of, player_names=player_names)
    write_players(matches, paths.artifacts / "players.parquet")
    write_match_history(matches, paths.artifacts / "match_history.parquet")
    log.info("artifacts written to %s", paths.artifacts)
    return 0


def run_publish(paths: Paths, version: str) -> int:
    log.warning("publish is a stub in Phase 1a; will copy artifacts to app-data in Phase 1b")
    _ = version
    _ = paths
    return 0


def main() -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(prog="progno-train")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("update_data")
    sub.add_parser("ingest")
    sub.add_parser("elo")
    publish = sub.add_parser("publish")
    publish.add_argument("version")
    args = parser.parse_args()

    paths = Paths.default(Path.cwd())

    if args.command == "update_data":
        return run_update_data(paths)
    if args.command == "ingest":
        return run_ingest(paths)
    if args.command == "elo":
        return run_elo(paths)
    if args.command == "publish":
        return run_publish(paths, args.version)
    return 1


if __name__ == "__main__":
    sys.exit(main())
