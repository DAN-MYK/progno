import numpy as np
import pytest

from progno_train.validate import acceptance_gate, compute_ece, compute_log_loss


def test_log_loss_perfect_prediction():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.999, 0.001, 0.999, 0.001])
    ll = compute_log_loss(y, p)
    assert ll < 0.01


def test_log_loss_random_prediction():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 1000)
    p = np.full(1000, 0.5)
    ll = compute_log_loss(y, p)
    assert abs(ll - np.log(2)) < 0.05  # ~0.693


def test_ece_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 1000)
    y = rng.binomial(1, p)
    ece = compute_ece(y, p)
    assert ece < 0.05  # Should be low for perfectly calibrated probs


def test_ece_miscalibrated():
    # Always predict 0.9 but true rate is 0.5
    y = np.array([1, 0] * 500)
    p = np.full(1000, 0.9)
    ece = compute_ece(y, p)
    assert ece > 0.3


def test_acceptance_gate_passes():
    # Should not raise
    acceptance_gate(model_logloss=0.60, baseline_logloss=0.65, ece=0.02)


def test_acceptance_gate_fails_logloss():
    with pytest.raises(ValueError, match="log-loss"):
        acceptance_gate(model_logloss=0.70, baseline_logloss=0.65, ece=0.02)


def test_acceptance_gate_fails_ece():
    with pytest.raises(ValueError, match="ECE"):
        acceptance_gate(model_logloss=0.60, baseline_logloss=0.65, ece=0.05)
