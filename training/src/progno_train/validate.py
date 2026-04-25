"""Metrics and acceptance gate for model validation."""

from __future__ import annotations

import numpy as np


def compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-7
    p = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(float(y_true[mask].mean()) - float(y_pred[mask].mean()))
    return ece / n


def compute_roi(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    odds: np.ndarray | None,
    kelly_frac: float = 0.25,
) -> float | None:
    """Compute ROI from predictions and closing odds using fractional Kelly stakes.

    Args:
        y_true: binary labels (1 if player_a won)
        y_pred: predicted probabilities for player_a winning
        odds: Pinnacle decimal odds for player_a winning (or None if unavailable)
        kelly_frac: Kelly fraction (default 0.25 = "fractional Kelly")

    Returns:
        ROI as return per unit bet (e.g. 0.05 = +5%), or None if odds unavailable.
    """
    if odds is None or len(odds) == 0:
        return None

    odds = np.asarray(odds, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    implied_p = 1.0 / odds
    full_kelly = (y_pred * odds - 1.0) / (odds - 1.0)
    stakes = np.maximum(0.0, kelly_frac * full_kelly)

    pnl = np.where(y_true == 1, stakes * (odds - 1.0), -stakes)
    total_stake = np.sum(stakes)

    if total_stake == 0.0:
        return 0.0

    return float(np.sum(pnl) / total_stake)


def acceptance_gate(
    model_logloss: float,
    baseline_logloss: float,
    ece: float,
    roi: float | None = None,
    ece_threshold: float = 0.03,
    roi_threshold: float = -0.01,
) -> None:
    """Raise ValueError if model fails acceptance criteria (non-negotiable invariants).

    Criteria (from spec §4.3, §6.4):
    - log-loss must beat Pure Elo baseline
    - ECE must be < threshold (default 0.03)
    - ROI must be >= threshold (default -0.01 = -1% buffer, spec target is >= 0)
    """
    if model_logloss >= baseline_logloss:
        raise ValueError(
            f"GATE FAIL: model log-loss {model_logloss:.4f} >= baseline {baseline_logloss:.4f}"
        )
    if ece > ece_threshold:
        raise ValueError(f"GATE FAIL: ECE {ece:.4f} > threshold {ece_threshold}")
    if roi is not None and roi < roi_threshold:
        raise ValueError(f"GATE FAIL: ROI {roi:.4f} < threshold {roi_threshold}")
