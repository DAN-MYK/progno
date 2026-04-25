import numpy as np
import pytest

from progno_train.validate import compute_log_loss, compute_ece, compute_roi, acceptance_gate


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


def test_compute_roi_none_when_odds_missing():
    assert compute_roi(np.array([1, 0]), np.array([0.6, 0.4]), None) is None


def test_compute_roi_handles_nan_odds():
    y = np.array([1, 0, 1])
    p = np.array([0.7, 0.4, 0.6])
    odds = np.array([2.0, np.nan, 1.8])
    roi = compute_roi(y, p, odds)
    assert roi is not None
    assert np.isfinite(roi)


def test_compute_roi_zero_stake_returns_zero():
    # implied prob (0.5) > model prob (0.4) → negative kelly → no stake
    y = np.array([1, 0])
    p = np.array([0.4, 0.4])
    odds = np.array([2.0, 2.0])
    roi = compute_roi(y, p, odds)
    assert roi == 0.0


def test_compute_roi_positive_when_model_beats_market():
    y = np.array([1, 1, 1, 1])
    p = np.array([0.9, 0.9, 0.9, 0.9])
    odds = np.array([2.0, 2.0, 2.0, 2.0])
    roi = compute_roi(y, p, odds)
    assert roi is not None
    assert roi > 0.5


def test_compute_roi_negative_when_model_wrong():
    y = np.array([0, 0, 0, 0])
    p = np.array([0.9, 0.9, 0.9, 0.9])
    odds = np.array([2.0, 2.0, 2.0, 2.0])
    roi = compute_roi(y, p, odds)
    assert roi is not None
    assert roi < -0.5


def test_compute_roi_invalid_odds_excluded():
    # odds <= 1.0 are masked; only valid match counted
    y = np.array([1, 1])
    p = np.array([0.7, 0.7])
    odds = np.array([0.5, 2.0])
    roi = compute_roi(y, p, odds)
    # stake = 0.25 * (0.7*2-1)/(2-1) = 0.1; win → pnl=0.1; ROI=1.0
    assert roi == pytest.approx(1.0)


def test_acceptance_gate_fails_on_negative_roi():
    with pytest.raises(ValueError, match="ROI"):
        acceptance_gate(model_logloss=0.6, baseline_logloss=0.65, ece=0.02, roi=-0.05)


def test_acceptance_gate_passes_on_borderline_roi():
    acceptance_gate(0.6, 0.65, 0.02, roi=-0.005)  # -0.5% >= -1% → PASS
    with pytest.raises(ValueError):
        acceptance_gate(0.6, 0.65, 0.02, roi=-0.02)  # -2% < -1% → FAIL


def test_acceptance_gate_keyword_only_args():
    # roi is keyword-only after *, so 4th positional must raise TypeError
    with pytest.raises(TypeError):
        acceptance_gate(0.6, 0.65, 0.02, 0.05)  # type: ignore[call-arg]
