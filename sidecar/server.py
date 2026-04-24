"""FastAPI sidecar — loads model artifacts, serves /health /predict /model_info."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from catboost import CatBoostClassifier, Pool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Global state set during startup ──────────────────────────────────────────
_model: CatBoostClassifier | None = None
_platt_a: float = 1.0
_platt_b: float = 0.0
_feature_cols: list[str] = []
_cat_idx: list[int] = []
_match_history: pd.DataFrame | None = None
_elo_state: dict = {}
_model_card: dict = {}
_port: int = 0

_CAT_FEATURES = {"surface", "tourney_level", "round"}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _load_artifacts(artifacts_dir: Path) -> None:
    global _model, _platt_a, _platt_b, _feature_cols, _cat_idx, _match_history, _elo_state, _model_card

    model_path = artifacts_dir / "model.cbm"
    if not model_path.exists():
        print(f"ERROR model.cbm not found at {model_path}", flush=True)
        sys.exit(1)

    for fname in ["calibration.json", "match_history.parquet", "elo_state.json"]:
        if not (artifacts_dir / fname).exists():
            print(f"ERROR {fname} not found at {artifacts_dir}", flush=True)
            sys.exit(1)

    _model = CatBoostClassifier()
    _model.load_model(str(model_path))
    _feature_cols = _model.feature_names_
    _cat_idx = [i for i, c in enumerate(_feature_cols) if c in _CAT_FEATURES]

    cal = json.loads((artifacts_dir / "calibration.json").read_text())
    _platt_a = cal["a"]
    _platt_b = cal["b"]

    _match_history = pd.read_parquet(artifacts_dir / "match_history.parquet")
    _elo_state = json.loads((artifacts_dir / "elo_state.json").read_text())

    card_path = artifacts_dir / "model_card.json"
    _model_card = json.loads(card_path.read_text()) if card_path.exists() else {}


def _apply_platt(raw: np.ndarray) -> np.ndarray:
    eps = 1e-7
    clipped = np.clip(raw, eps, 1 - eps)
    logits = np.log(clipped) - np.log(1 - clipped)
    return 1.0 / (1.0 + np.exp(-(_platt_a * logits + _platt_b)))


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    sys.stdout.write(f"READY port={_port}\n")
    sys.stdout.flush()
    yield


app = FastAPI(lifespan=lifespan)


class MatchRequest(BaseModel):
    player_a_id: str
    player_b_id: str
    surface: str
    tourney_level: str = "A"
    round_: str = "R32"
    best_of: int = 3
    tourney_date: str  # "YYYY-MM-DD"


class PredictRequest(BaseModel):
    matches: list[MatchRequest]


class MatchPrediction(BaseModel):
    prob_a_wins: float
    prob_a_wins_uncalibrated: float
    elo_prob_a_wins: float
    confidence_flag: str  # "ok" | "low_history" | "insufficient_data"


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[MatchPrediction]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/model_info")
async def model_info():
    return _model_card


@app.post("/predict")
async def predict(req: PredictRequest) -> PredictResponse:
    from progno_train.features import compute_match_features

    if _model is None or _match_history is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = []
    for m in req.matches:
        tourney_date = pd.Timestamp(m.tourney_date)
        try:
            pid_a = int(m.player_a_id)
            pid_b = int(m.player_b_id)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"player IDs must be integers: {m.player_a_id}, {m.player_b_id}"
            )

        try:
            feats = compute_match_features(
                history=_match_history,
                elo_state=_elo_state,
                player_a_id=pid_a,
                player_b_id=pid_b,
                surface=m.surface,
                tourney_level=m.tourney_level,
                round_=m.round_,
                best_of=m.best_of,
                tourney_date=tourney_date,
            )

            low_history = bool(feats.get("low_history_flag", 0))
            conf_flag = "low_history" if low_history else "ok"

            feat_row = {col: feats.get(col, 0) for col in _feature_cols}
            feat_df = pd.DataFrame([feat_row])
            pool = Pool(feat_df, cat_features=_cat_idx, feature_names=_feature_cols)

            raw = float(_model.predict_proba(pool)[0, 1])
            cal = float(_apply_platt(np.array([raw]))[0])

            # Elo baseline from feature (elo_overall_diff encodes rating difference)
            elo_diff = feats.get("elo_overall_diff", 0.0)
            elo_prob = float(1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0)))

            predictions.append(MatchPrediction(
                prob_a_wins=round(cal, 4),
                prob_a_wins_uncalibrated=round(raw, 4),
                elo_prob_a_wins=round(elo_prob, 4),
                confidence_flag=conf_flag,
            ))
        except Exception as exc:
            import logging
            logging.getLogger("progno_sidecar").warning(
                "inference failed for player_a=%s player_b=%s: %s",
                m.player_a_id, m.player_b_id, exc
            )
            predictions.append(MatchPrediction(
                prob_a_wins=0.5,
                prob_a_wins_uncalibrated=0.5,
                elo_prob_a_wins=0.5,
                confidence_flag="error",
            ))

    model_version = _model_card.get("generated_at", "unknown")
    return PredictResponse(model_version=model_version, predictions=predictions)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global _port
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True)
    args = parser.parse_args()

    _load_artifacts(Path(args.artifacts_dir))
    _port = _find_free_port()
    uvicorn.run(app, host="127.0.0.1", port=_port, log_level="error")


if __name__ == "__main__":
    main()
