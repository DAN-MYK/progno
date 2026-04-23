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
    out = paths.matches_clean
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("wrote %d rows to %s", len(df), out)
    return 0


def run_elo(paths: Paths) -> int:
    staging = paths.matches_clean
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
    write_elo_state(state, paths.elo_state, data_as_of=data_as_of, player_names=player_names)
    write_players(matches, paths.players)
    write_match_history(matches, paths.match_history)
    log.info("artifacts written to %s", paths.artifacts)
    return 0


def run_publish(paths: Paths, version: str) -> int:
    log.warning("publish is a stub in Phase 1a; will copy artifacts to app-data in Phase 1b")
    _ = version
    _ = paths
    return 0


def run_features(paths: Paths) -> None:
    import json

    from progno_train.features import build_all_features

    if not paths.match_history.exists():
        log.error("match_history.parquet not found at %s — run 'just elo' first", paths.match_history)
        raise SystemExit(2)
    if not paths.elo_state.exists():
        log.error("elo_state.json not found at %s — run 'just elo' first", paths.elo_state)
        raise SystemExit(2)

    log.info("Loading match history for feature engineering...")
    history = pd.read_parquet(paths.match_history)
    elo_state = json.loads(paths.elo_state.read_text())

    log.info("Building features for %d matches...", len(history))
    featurized = build_all_features(history, elo_state)
    paths.featurized.parent.mkdir(parents=True, exist_ok=True)
    featurized.to_parquet(paths.featurized, index=False)
    log.info("Featurized dataset written: %s (%d rows)", paths.featurized, len(featurized))


def run_train(paths: Paths) -> None:
    import subprocess

    from progno_train.artifacts import write_calibration, write_model_card
    from progno_train.train import run_walk_forward

    log.info("Running walk-forward training...")
    model, a, b, metrics, feature_cols = run_walk_forward(paths.featurized)

    log.info("Saving model artifacts...")
    model.save_model(str(paths.model_cbm))
    write_calibration(a, b, paths.calibration)

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], timeout=5).decode().strip()
    except Exception:
        git_sha = "unknown"

    write_model_card(
        train_years=(2005, 2022),
        test_year=2023,
        metrics=metrics,
        feature_names=feature_cols,
        git_sha=git_sha,
        out_path=paths.model_card,
    )
    log.info("Training complete. Metrics: %s", metrics)


def run_validate(paths: Paths) -> None:
    import json

    import numpy as np
    from catboost import CatBoostClassifier, Pool

    from progno_train.train import apply_platt, get_feature_cols
    from progno_train.validate import acceptance_gate, compute_ece, compute_log_loss

    if not paths.model_cbm.exists():
        log.error("model.cbm not found at %s — run 'just train' first", paths.model_cbm)
        raise SystemExit(2)
    if not paths.calibration.exists():
        log.error("calibration.json not found at %s — run 'just train' first", paths.calibration)
        raise SystemExit(2)
    if not paths.featurized.exists():
        log.error("featurized dataset not found at %s — run 'just features' first", paths.featurized)
        raise SystemExit(2)

    log.info("Running validation and acceptance gate...")
    model = CatBoostClassifier()
    model.load_model(str(paths.model_cbm))

    cal = json.loads(paths.calibration.read_text())
    a, b = cal["a"], cal["b"]

    df = pd.read_parquet(paths.featurized)
    test_df = df[df["year"] >= 2023]
    feature_cols = get_feature_cols(df)

    pool = Pool(test_df[feature_cols].fillna(0), feature_names=feature_cols)
    raw = model.predict_proba(pool)[:, 1]
    cal_probs = apply_platt(raw, a, b)

    y = test_df["label"].values
    elo_probs = 1.0 / (1.0 + 10.0 ** (-test_df["elo_overall_diff"].values / 400.0))

    model_ll = compute_log_loss(y, cal_probs)
    baseline_ll = compute_log_loss(y, elo_probs)
    ece = compute_ece(y, cal_probs)

    log.info("Model log-loss: %.4f | Elo baseline: %.4f | ECE: %.4f", model_ll, baseline_ll, ece)
    acceptance_gate(model_ll, baseline_ll, ece)
    log.info("Acceptance gate: PASS")


def run_retrain(paths: Paths, version: str) -> None:
    log.info("Starting retrain pipeline...")
    run_features(paths)
    run_train(paths)
    run_validate(paths)
    run_publish(paths, version)
    log.info("Retrain pipeline complete")


def main() -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(prog="progno-train")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("update_data")
    sub.add_parser("ingest")
    sub.add_parser("elo")
    sub.add_parser("features")
    sub.add_parser("train")
    sub.add_parser("validate")
    retrain = sub.add_parser("retrain")
    retrain.add_argument("--version", default="dev")
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
    if args.command == "features":
        run_features(paths)
        return 0
    if args.command == "train":
        run_train(paths)
        return 0
    if args.command == "validate":
        run_validate(paths)
        return 0
    if args.command == "retrain":
        run_retrain(paths, args.version)
        return 0
    if args.command == "publish":
        return run_publish(paths, args.version)
    return 1


if __name__ == "__main__":
    sys.exit(main())
