"""CLI entry points for the training pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from progno_train.config import Paths

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


def _require(path: Path, hint: str) -> int | None:
    """Return 2 (error exit code) if path does not exist, else None."""
    if not path.exists():
        log.error("%s not found at %s", hint, path)
        return 2
    return None


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

    # Merge supplemental tennis-data.co.uk XLSX if available
    xlsx_dir = paths.data_raw / "tennis_data_xlsx"
    if xlsx_dir.exists():
        from progno_train.ingest_xlsx import ingest_tennis_data_xlsx_dir
        players_path = paths.players
        players = pd.read_parquet(players_path) if players_path.exists() else None
        xl_df = ingest_tennis_data_xlsx_dir(xlsx_dir, players=players, tour=tour)
        if not xl_df.empty:
            max_sackmann = df["tourney_date"].max()
            xl_new = xl_df[xl_df["tourney_date"] > max_sackmann]
            if not xl_new.empty:
                log.info(
                    "merging %d supplemental rows from tennis-data.co.uk (after %s)",
                    len(xl_new), max_sackmann.date(),
                )
                df = pd.concat([df, xl_new], ignore_index=True)
                df = df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)

    out = paths.matches_clean
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("wrote %d rows to %s", len(df), out)
    return 0


def run_elo(paths: Paths) -> int:
    if (rc := _require(paths.matches_clean, "staging parquet (run ingest first)")) is not None:
        return rc
    matches = pd.read_parquet(paths.matches_clean)
    log.info("rolling up %d matches", len(matches))
    state = rollup_elo(matches)
    log.info("produced Elo state for %d players", len(state))

    data_as_of = matches["tourney_date"].max()
    all_names = pd.concat([
        matches[["winner_id", "winner_name"]].rename(columns={"winner_id": "id", "winner_name": "name"}),
        matches[["loser_id", "loser_name"]].rename(columns={"loser_id": "id", "loser_name": "name"}),
    ]).drop_duplicates("id")
    player_names = {
        int(row.id): parts[-1].lower()
        for row in all_names.itertuples()
        if (parts := str(row.name).split())
    }

    # Phase 3.5: join closing odds from tennis-data.co.uk XLSX if available
    xlsx_files = sorted(paths.odds_xlsx_dir.glob("*.xlsx")) if paths.odds_xlsx_dir.exists() else []
    if xlsx_files:
        from progno_train.ingest_xlsx import ingest_tennis_data_xlsx
        from progno_train.odds_join import join_odds
        log.info("joining odds from %d XLSX files...", len(xlsx_files))
        odds_df = ingest_tennis_data_xlsx(xlsx_files)
        matches = join_odds(matches, odds_df, name_map_path=paths.name_map)
        log.info("odds joined — PSW coverage: %.1f%%",
                 100.0 * matches["PSW"].notna().mean())
    else:
        log.warning("no XLSX files in %s — ROI gate will be skipped", paths.odds_xlsx_dir)

    paths.artifacts.mkdir(parents=True, exist_ok=True)
    write_elo_state(state, paths.elo_state, data_as_of=data_as_of, player_names=player_names)
    write_players(matches, paths.players)
    write_match_history(matches, paths.match_history)
    log.info("artifacts written to %s", paths.artifacts)
    return 0


def run_features(paths: Paths, tour: str = "atp") -> int:
    from progno_train.features import build_all_features
    from progno_train.train import BURN_IN_YEAR_ATP, BURN_IN_YEAR_WTA

    if (rc := _require(paths.match_history, "match_history (run elo first)")) is not None:
        return rc

    min_year = BURN_IN_YEAR_ATP if tour == "atp" else BURN_IN_YEAR_WTA

    log.info("loading match history for feature engineering...")
    history = pd.read_parquet(paths.match_history)
    elo_state = json.loads(paths.elo_state.read_text())

    # Remap elo_state from last-name keys to str(int_pid) keys so _elo(int_pid, field) works.
    # elo_state.json is keyed by last_name (for Rust lookup); features.py does str(int_pid) lookup.
    if paths.players.exists():
        _players_df = pd.read_parquet(paths.players)
        _pid_to_last: dict[int, str] = {}
        for _row in _players_df.itertuples():
            _parts = str(_row.name).split()
            if _parts:
                _pid_to_last[int(_row.player_id)] = _parts[-1].lower()
        _by_name = elo_state.get("players", {})
        elo_state = {
            **elo_state,
            "players": {
                str(pid): _by_name[last]
                for pid, last in _pid_to_last.items()
                if last in _by_name
            },
        }

    log.info("building features for %d matches (min_year=%d)...", len(history), min_year)
    featurized = build_all_features(history, elo_state, min_year=min_year)
    paths.featurized.parent.mkdir(parents=True, exist_ok=True)
    featurized.to_parquet(paths.featurized, index=False)
    log.info("featurized dataset written: %s (%d rows)", paths.featurized, len(featurized))
    return 0


def run_train(paths: Paths, tour: str) -> int:
    from progno_train.train import run_walk_forward, BURN_IN_YEAR_ATP, BURN_IN_YEAR_WTA

    if (rc := _require(paths.featurized, "featurized parquet (run features first)")) is not None:
        return rc

    burn_in = BURN_IN_YEAR_WTA if tour == "wta" else BURN_IN_YEAR_ATP
    val_start = 2019 if tour == "wta" else 2016
    log.info("running walk-forward training (tour=%s, burn_in=%d, val_start=%d)...", tour, burn_in, val_start)
    model, a, b, metrics, feature_cols = run_walk_forward(paths.featurized, burn_in_year=burn_in, val_start=val_start)

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


def _extract_winner_odds(test_df: pd.DataFrame) -> tuple[object, str | None]:
    """Find first available 'odds for winner' column from tennis-data.co.uk.

    Returns (odds_array, column_name) or (None, None) if none found.
    Note: these columns land in match_history only after ingest_xlsx propagates them (Phase 3.5).
    """
    candidates = ["odds_a_winner", "PSW", "B365W", "MaxW", "AvgW"]
    for col in candidates:
        if col in test_df.columns and test_df[col].notna().any():
            return test_df[col].to_numpy(), col
    return None, None


def run_validate(paths: Paths) -> int:
    from catboost import CatBoostClassifier, Pool
    from progno_train.train import apply_platt, get_feature_cols, TEST_START_YEAR
    from progno_train.validate import compute_log_loss, compute_ece, compute_roi, acceptance_gate

    log.info("running validation and acceptance gate...")
    model = CatBoostClassifier()
    model.load_model(str(paths.model_cbm))

    cal = json.loads(paths.calibration.read_text())
    a, b = cal["a"], cal["b"]

    df = pd.read_parquet(paths.featurized)
    test_df = df[df["year"] >= TEST_START_YEAR]
    feature_cols = get_feature_cols(df)

    from progno_train.train import CAT_FEATURES
    cat_idx = [i for i, c in enumerate(feature_cols) if c in CAT_FEATURES]
    pool = Pool(test_df[feature_cols].fillna(0), cat_features=cat_idx, feature_names=feature_cols)
    raw = model.predict_proba(pool)[:, 1]
    cal_probs = apply_platt(raw, a, b)

    y = test_df["label"].values
    elo_probs = (1.0 / (1.0 + 10 ** (-test_df["elo_overall_diff"].values / 400))).clip(0.05, 0.95)

    model_ll = compute_log_loss(y, cal_probs)
    baseline_ll = compute_log_loss(y, elo_probs)
    ece = compute_ece(y, cal_probs)

    odds_arr, odds_col = _extract_winner_odds(test_df)
    roi = None
    if odds_arr is not None:
        roi = compute_roi(y, cal_probs, odds_arr)
        log.info("ROI (0.25× Kelly via %s): %.4f", odds_col, roi or 0.0)
    else:
        log.warning(
            "ROI gate SKIPPED: no odds column found in test data "
            "(expected one of: odds_a_winner, PSW, B365W, MaxW, AvgW). "
            "Odds ingest from tennis-data.co.uk is Phase 3.5 work."
        )

    log.info("model log-loss: %.4f | Elo baseline: %.4f | ECE: %.4f", model_ll, baseline_ll, ece)
    try:
        acceptance_gate(model_ll, baseline_ll, ece, roi=roi)
        log.info("acceptance gate: PASS")
    except ValueError as e:
        log.error("acceptance gate: FAIL — %s", e)
        return 1

    # Generate report.md alongside the model artifacts (§6.6)
    try:
        from progno_train.report import generate_report
        report_path = generate_report(paths, tour=_infer_tour(paths))
        log.info("report written to %s", report_path)
    except Exception as exc:
        log.warning("report generation skipped: %s", exc)

    return 0


def _infer_tour(paths: "Paths") -> str:
    """Derive tour from the artifact dir name (atp / wta)."""
    return paths.artifacts.name if paths.artifacts.name in ("atp", "wta") else "atp"


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
    import shutil
    src = paths.artifacts
    if not src.exists():
        log.error("no artifacts at %s — run retrain first", src)
        return 2

    versioned = src.parent / f"v{version}"
    versioned.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, versioned / f.name)

    current = src.parent / "current"
    if current.is_symlink() or current.exists():
        current.unlink()
    current.symlink_to(versioned.name)

    log.info("published %s → %s", src.name, versioned)
    log.info("current symlink: %s → %s", current, versioned.name)
    log.info("copy '%s/' to app artifacts/%s/ to activate", versioned, src.name)
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
    sub.add_parser("report")
    sub.add_parser("retrain").add_argument("--version", default="dev")
    sub.add_parser("publish").add_argument("version")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent  # = training/
    paths = Paths.for_tour(root, args.tour)

    def _run_report() -> int:
        from progno_train.report import generate_report
        p = generate_report(paths, tour=args.tour)
        log.info("report written to %s", p)
        return 0

    dispatch = {
        "update_data": lambda: run_update_data(paths),
        "ingest": lambda: run_ingest(paths, args.tour),
        "elo": lambda: run_elo(paths),
        "features": lambda: run_features(paths, args.tour),
        "train": lambda: run_train(paths, args.tour),
        "validate": lambda: run_validate(paths),
        "report": _run_report,
        "retrain": lambda: run_retrain(paths, args.tour, getattr(args, "version", "dev")),
        "publish": lambda: run_publish(paths, getattr(args, "version", "dev")),
    }
    return dispatch[args.command]()


if __name__ == "__main__":
    sys.exit(main())
