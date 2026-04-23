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
    def default(cls, root: Path) -> Paths:
        return cls(
            data_raw=root / "data" / "raw",
            data_staging=root / "data" / "staging",
            artifacts=root / "artifacts",
        )
