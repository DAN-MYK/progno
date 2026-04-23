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
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_pred[mask].mean())
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / n


def acceptance_gate(
    model_logloss: float,
    baseline_logloss: float,
    ece: float,
    ece_threshold: float = 0.03,
) -> None:
    """Raise ValueError if model fails acceptance criteria."""
    if model_logloss >= baseline_logloss:
        raise ValueError(
            f"GATE FAIL: model log-loss {model_logloss:.4f} >= baseline {baseline_logloss:.4f}"
        )
    if ece > ece_threshold:
        raise ValueError(
            f"GATE FAIL: ECE {ece:.4f} > threshold {ece_threshold}"
        )


def elo_baseline_logloss(y_true: np.ndarray, elo_probs: np.ndarray) -> float:
    return compute_log_loss(y_true, elo_probs)
