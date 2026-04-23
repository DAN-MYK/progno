"""Path configuration for the training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data_raw: Path
    data_staging: Path
    artifacts: Path

    @classmethod
    def default(cls, root: Path) -> "Paths":
        return cls(
            data_raw=root / "data" / "raw",
            data_staging=root / "data" / "staging",
            artifacts=root / "artifacts",
        )

    @classmethod
    def for_tour(cls, root: Path, tour: str) -> "Paths":
        return cls(
            data_raw=root / "data" / "raw",
            data_staging=root / "data" / "staging" / tour,
            artifacts=root / "artifacts" / tour,
        )

    @property
    def matches_raw(self) -> Path:
        return self.data_staging / "matches_raw.parquet"

    @property
    def matches_clean(self) -> Path:
        return self.data_staging / "matches_clean.parquet"

    @property
    def featurized(self) -> Path:
        return self.data_staging / "matches_featurized.parquet"

    @property
    def match_history(self) -> Path:
        return self.artifacts / "match_history.parquet"

    @property
    def elo_state(self) -> Path:
        return self.artifacts / "elo_state.json"

    @property
    def players(self) -> Path:
        return self.artifacts / "players.parquet"

    @property
    def model_cbm(self) -> Path:
        return self.artifacts / "model.cbm"

    @property
    def calibration(self) -> Path:
        return self.artifacts / "calibration.json"

    @property
    def model_card(self) -> Path:
        return self.artifacts / "model_card.json"
