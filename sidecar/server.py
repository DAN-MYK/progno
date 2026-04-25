"""FastAPI sidecar — loads ATP and WTA models, routes by tour field."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from catboost import CatBoostClassifier, Pool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import features as feat_module

_models: dict[str, CatBoostClassifier | None] = {"atp": None, "wta": None}
_platt: dict[str, tuple[float, float]] = {"atp": (1.0, 0.0), "wta": (1.0, 0.0)}
_cat_idx: dict[str, list[int]] = {"atp": [], "wta": []}
_history: dict[str, pd.DataFrame | None] = {"atp": None, "wta": None}
_elo_state: dict[str, dict] = {"atp": {}, "wta": {}}
# last_name -> integer player_id; keyed the same way Rust normalises names (last word, lowercase)
_name_to_pid: dict[str, dict[str, int]] = {"atp": {}, "wta": {}}
_model_card: dict[str, dict] = {"atp": {}, "wta": {}}
_port: int = 0

_CAT_FEATURES = {"surface", "tourney_level", "round"}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _load_tour(artifacts_root: Path, tour: str) -> None:
    tour_dir = artifacts_root / tour
    model_path = tour_dir / "model.cbm"
    if not model_path.exists():
        print(f"INFO no model for {tour} at {model_path} — skipping", flush=True)
        return

    m = CatBoostClassifier()
    m.load_model(str(model_path))
    _models[tour] = m
    _cat_idx[tour] = [i for i, c in enumerate(m.feature_names_) if c in _CAT_FEATURES]

    cal = json.loads((tour_dir / "calibration.json").read_text())
    _platt[tour] = (cal["a"], cal["b"])

    _history[tour] = pd.read_parquet(tour_dir / "match_history.parquet")
    elo_raw = json.loads((tour_dir / "elo_state.json").read_text())

    # Build name->pid lookup and remap elo_state from last-name keys to str(int_pid) keys.
    # elo_state.json uses last-name keys (for Rust lookup); features.py does str(int_pid) lookup.
    players_path = tour_dir / "players.parquet"
    if players_path.exists():
        players_df = pd.read_parquet(players_path)
        pid_to_last: dict[int, str] = {}
        for row in players_df.itertuples():
            parts = str(row.name).split()
            if parts:
                pid_to_last[int(row.player_id)] = parts[-1].lower()
        _name_to_pid[tour] = {last: pid for pid, last in pid_to_last.items()}
        players_by_name = elo_raw.get("players", {})
        _elo_state[tour] = {
            **elo_raw,
            "players": {
                str(pid): players_by_name[last]
                for pid, last in pid_to_last.items()
                if last in players_by_name
            },
        }
    else:
        _elo_state[tour] = elo_raw

    card_path = tour_dir / "model_card.json"
    _model_card[tour] = json.loads(card_path.read_text()) if card_path.exists() else {}

    print(f"INFO loaded {tour} model", flush=True)


def _apply_platt(raw: np.ndarray, tour: str) -> np.ndarray:
    a, b = _platt[tour]
    eps = 1e-7
    clipped = np.clip(raw, eps, 1 - eps)
    logits = np.log(clipped) - np.log(1 - clipped)
    return 1.0 / (1.0 + np.exp(-(a * logits + b)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    sys.stdout.write(f"READY port={_port}\n")
    sys.stdout.flush()
    yield


app = FastAPI(lifespan=lifespan)


class MatchRequest(BaseModel):
    tour: str
    player_a_id: str
    player_b_id: str
    surface: str
    tourney_level: str = "A"
    round_: str = "R32"
    best_of: int = 3
    tourney_date: str


class PredictRequest(BaseModel):
    matches: list[MatchRequest]


class MatchPrediction(BaseModel):
    prob_a_wins: float
    prob_a_wins_uncalibrated: float
    elo_prob_a_wins: float
    confidence_flag: str


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[MatchPrediction]


@app.get("/health")
async def health():
    return {"status": "ok", "tours_loaded": [t for t, m in _models.items() if m is not None]}


@app.get("/model_info")
async def model_info():
    return {
        t: {"loaded": _models[t] is not None, "card": _model_card.get(t, {})}
        for t in ("atp", "wta")
    }


@app.post("/predict")
async def predict(req: PredictRequest) -> PredictResponse:
    results = []
    for m in req.matches:
        tour = m.tour
        if _models.get(tour) is None:
            raise HTTPException(503, f"Model not loaded for tour: {tour}")

        # Rust sends last-name strings (e.g. "alcaraz"); map to integer player IDs.
        pid_a = _name_to_pid[tour].get(m.player_a_id, 0)
        pid_b = _name_to_pid[tour].get(m.player_b_id, 0)

        tourney_date = pd.Timestamp(m.tourney_date)
        feats = feat_module.compute_match_features(
            history=_history[tour],
            elo_state=_elo_state[tour],
            player_a_id=pid_a,
            player_b_id=pid_b,
            surface=m.surface,
            tourney_level=m.tourney_level,
            round_=m.round_,
            best_of=m.best_of,
            tourney_date=tourney_date,
        )

        low_history = bool(feats.get("low_history_flag", 0))
        feature_cols = _models[tour].feature_names_
        feat_df = pd.DataFrame([{col: feats.get(col, 0) for col in feature_cols}])
        pool = Pool(feat_df, cat_features=_cat_idx[tour], feature_names=list(feature_cols))

        raw = float(_models[tour].predict_proba(pool)[0, 1])
        cal = float(_apply_platt(np.array([raw]), tour)[0])
        elo_diff = feats.get("elo_overall_diff", 0.0)
        elo_prob = float(1.0 / (1.0 + 10 ** (-elo_diff / 400)))

        results.append(MatchPrediction(
            prob_a_wins=round(cal, 4),
            prob_a_wins_uncalibrated=round(raw, 4),
            elo_prob_a_wins=round(elo_prob, 4),
            confidence_flag="low_history" if low_history else "ok",
        ))

    version = _model_card.get(req.matches[0].tour, {}).get("generated_at", "unknown") if req.matches else "unknown"
    return PredictResponse(model_version=version, predictions=results)


def main() -> None:
    global _port
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts_root)
    for tour in ("atp", "wta"):
        _load_tour(root, tour)

    _port = _find_free_port()
    uvicorn.run(app, host="127.0.0.1", port=_port, log_level="error")


if __name__ == "__main__":
    main()
