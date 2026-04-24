import numpy as np
import pytest

from progno_train.validate import compute_log_loss, compute_ece, acceptance_gate


def test_log_loss_perfect():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.999, 0.001, 0.999, 0.001])
    assert compute_log_loss(y, p) < 0.01


def test_log_loss_random():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 1000)
    p = np.full(1000, 0.5)
    assert abs(compute_log_loss(y, p) - np.log(2)) < 0.05


def test_ece_well_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 1000)
    y = rng.binomial(1, p)
    assert compute_ece(y, p) < 0.05


def test_ece_miscalibrated():
    y = np.array([1, 0] * 500)
    p = np.full(1000, 0.9)
    assert compute_ece(y, p) > 0.3


def test_gate_passes():
    acceptance_gate(model_logloss=0.60, baseline_logloss=0.65, ece=0.02)


def test_gate_fails_logloss():
    with pytest.raises(ValueError, match="log-loss"):
        acceptance_gate(model_logloss=0.70, baseline_logloss=0.65, ece=0.02)


def test_gate_fails_ece():
    with pytest.raises(ValueError, match="ECE"):
        acceptance_gate(model_logloss=0.60, baseline_logloss=0.65, ece=0.05)
