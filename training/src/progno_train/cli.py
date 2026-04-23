"""CLI entry points for the training pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from progno_train.artifacts import (
    write_calibration,
    write_elo_state,
    write_match_history,
    write_model_card,
    write_players,
)
from progno_train.config import Paths
from progno_train.ingest import ingest_sackmann_csv
from progno_train.rollup import rollup_elo

log = logging.getLogger("progno_train")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def run_update_data(paths: Paths) -> int:
    log.info("update_data: pull latest Sackmann data")
    if not paths.data_raw.exists():
        paths.data_raw.mkdir(parents=True, exist_ok=True)
    return 0


def run_ingest(paths: Paths, tour: str) -> int:
    csv_glob = f"tennis_{tour}/{tour}_matches_*.csv"
    csvs = sorted(paths.data_raw.glob(csv_glob))
    if not csvs:
        log.error("no Sackmann CSVs found: %s/%s", paths.data_raw, csv_glob)
        return 2
    log.info("ingesting %d CSV files for tour=%s", len(csvs), tour)
    df = ingest_sackmann_csv(csvs)
    out = paths.matches_clean
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("wrote %d rows to %s", len(df), out)
    return 0


def run_elo(paths: Paths) -> int:
    if not paths.matches_clean.exists():
        log.error("no staging parquet at %s; run ingest first", paths.matches_clean)
        return 2
    matches = pd.read_parquet(paths.matches_clean)
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


def run_features(paths: Paths) -> int:
    from progno_train.features import build_all_features

    if not paths.match_history.exists():
        log.error("no match_history at %s; run elo first", paths.match_history)
        return 2

    log.info("loading match history for feature engineering...")
    history = pd.read_parquet(paths.match_history)
    elo_state = json.loads(paths.elo_state.read_text())

    log.info("building features for %d matches...", len(history))
    featurized = build_all_features(history, elo_state)
    paths.featurized.parent.mkdir(parents=True, exist_ok=True)
    featurized.to_parquet(paths.featurized, index=False)
    log.info("featurized dataset written: %s (%d rows)", paths.featurized, len(featurized))
    return 0


def run_train(paths: Paths, tour: str) -> int:
    from progno_train.train import run_walk_forward, BURN_IN_YEAR_ATP, BURN_IN_YEAR_WTA

    if not paths.featurized.exists():
        log.error("no featurized parquet at %s; run features first", paths.featurized)
        return 2

    burn_in = BURN_IN_YEAR_WTA if tour == "wta" else BURN_IN_YEAR_ATP
    log.info("running walk-forward training (tour=%s, burn_in=%d)...", tour, burn_in)
    model, a, b, metrics, feature_cols = run_walk_forward(paths.featurized, burn_in_year=burn_in)

    log.info("saving model artifacts...")
    model.save_model(str(paths.model_cbm))
    write_calibration(a, b, paths.calibration)
    write_model_card(
        train_years=(burn_in + 1, metrics.get("cal_year", 2022)),
        test_year=metrics.get("test_year", 2023),
        metrics=metrics,
        feature_names=feature_cols,
        git_sha=_git_sha(),
        out_path=paths.model_card,
    )
    log.info("training complete. metrics: %s", metrics)
    return 0


def run_validate(paths: Paths) -> int:
    from catboost import CatBoostClassifier, Pool
    from progno_train.train import apply_platt, get_feature_cols
    from progno_train.validate import compute_log_loss, compute_ece, acceptance_gate

    log.info("running validation and acceptance gate...")
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
    elo_probs = (1.0 / (1.0 + 10 ** (-test_df["elo_overall_diff"].values / 400))).clip(0.05, 0.95)

    model_ll = compute_log_loss(y, cal_probs)
    baseline_ll = compute_log_loss(y, elo_probs)
    ece = compute_ece(y, cal_probs)

    log.info("model log-loss: %.4f | Elo baseline: %.4f | ECE: %.4f", model_ll, baseline_ll, ece)
    try:
        acceptance_gate(model_ll, baseline_ll, ece)
        log.info("acceptance gate: PASS")
    except ValueError as e:
        log.error("acceptance gate: FAIL — %s", e)
        return 1
    return 0


def run_retrain(paths: Paths, tour: str, version: str) -> int:
    for fn in (
        lambda: run_ingest(paths, tour),
        lambda: run_elo(paths),
        lambda: run_features(paths),
        lambda: run_train(paths, tour),
        lambda: run_validate(paths),
    ):
        rc = fn()
        if rc != 0:
            return rc
    log.info("retrain complete for tour=%s version=%s", tour, version)
    return 0


def run_publish(paths: Paths, version: str) -> int:
    log.warning("publish: stub — copy artifacts/%s to app-data in Phase 5", paths.artifacts.name)
    return 0


def main() -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(prog="progno-train")
    parser.add_argument("--tour", choices=["atp", "wta"], default="atp",
                        help="Tour to process (default: atp)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("update_data")
    sub.add_parser("ingest")
    sub.add_parser("elo")
    sub.add_parser("features")
    sub.add_parser("train")
    sub.add_parser("validate")
    sub.add_parser("retrain").add_argument("--version", default="dev")
    sub.add_parser("publish").add_argument("version")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent  # = training/
    paths = Paths.for_tour(root, args.tour)

    dispatch = {
        "update_data": lambda: run_update_data(paths),
        "ingest": lambda: run_ingest(paths, args.tour),
        "elo": lambda: run_elo(paths),
        "features": lambda: run_features(paths),
        "train": lambda: run_train(paths, args.tour),
        "validate": lambda: run_validate(paths),
        "retrain": lambda: run_retrain(paths, args.tour, getattr(args, "version", "dev")),
        "publish": lambda: run_publish(paths, getattr(args, "version", "dev")),
    }
    return dispatch[args.command]()


if __name__ == "__main__":
    sys.exit(main())
