"""Tests for evaluator — metrics correctness for classification and regression."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.evaluator import evaluate_pipeline, champion_metric, full_report


# ── Helpers ────────────────────────────────────────────────────

def _make_clf_pipeline(y_true, accuracy=0.8):
    """Creates a mock pipeline that predicts with given accuracy."""
    n      = len(y_true)
    n_correct = int(n * accuracy)
    y_pred = y_true.copy()
    # Flip some predictions to control accuracy
    flip_idx = np.arange(n - n_correct)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]

    mock = MagicMock()
    mock.predict.return_value = y_pred
    mock.predict_proba.return_value = np.column_stack([1 - y_pred * 0.8, y_pred * 0.8 + 0.1])
    return mock


def _make_reg_pipeline(y_true, noise=0.1):
    """Creates a mock pipeline with small prediction noise."""
    rng    = np.random.default_rng(0)
    y_pred = y_true + rng.normal(0, noise * np.std(y_true), len(y_true))
    mock   = MagicMock()
    mock.predict.return_value = y_pred
    return mock


# ── Classification metrics ─────────────────────────────────────

class TestClassificationMetrics:

    @pytest.fixture
    def clf_data(self):
        rng    = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        X_val  = pd.DataFrame({"f": rng.normal(0, 1, 200)})
        return X_val, y_true

    def test_returns_required_keys(self, clf_data):
        X_val, y_true = clf_data
        pipe    = _make_clf_pipeline(y_true)
        metrics = evaluate_pipeline(pipe, X_val, y_true, task="classification")
        for key in ["val_accuracy", "val_f1", "val_roc_auc", "val_precision", "val_recall"]:
            assert key in metrics

    def test_accuracy_range(self, clf_data):
        X_val, y_true = clf_data
        pipe    = _make_clf_pipeline(y_true, accuracy=0.75)
        metrics = evaluate_pipeline(pipe, X_val, y_true, task="classification")
        assert 0.0 <= metrics["val_accuracy"] <= 1.0

    def test_all_metrics_are_float(self, clf_data):
        X_val, y_true = clf_data
        pipe    = _make_clf_pipeline(y_true)
        metrics = evaluate_pipeline(pipe, X_val, y_true, task="classification")
        for v in metrics.values():
            assert isinstance(v, float)

    def test_roc_auc_range(self, clf_data):
        X_val, y_true = clf_data
        pipe    = _make_clf_pipeline(y_true)
        metrics = evaluate_pipeline(pipe, X_val, y_true, task="classification")
        assert 0.0 <= metrics["val_roc_auc"] <= 1.0


# ── Regression metrics ─────────────────────────────────────────

class TestRegressionMetrics:

    @pytest.fixture
    def reg_data(self):
        rng    = np.random.default_rng(42)
        y_true = rng.uniform(100_000, 900_000, 200)
        X_val  = pd.DataFrame({"f": rng.normal(0, 1, 200)})
        return X_val, y_true

    def test_returns_required_keys(self, reg_data):
        X_val, y_true = reg_data
        pipe    = _make_reg_pipeline(y_true)
        metrics = evaluate_pipeline(pipe, X_val, y_true, task="regression")
        for key in ["val_rmse", "val_mae", "val_r2", "val_mape"]:
            assert key in metrics

    def test_rmse_is_positive(self, reg_data):
        X_val, y_true = reg_data
        pipe    = _make_reg_pipeline(y_true, noise=0.05)
        metrics = evaluate_pipeline(pipe, X_val, y_true, task="regression")
        assert metrics["val_rmse"] >= 0

    def test_r2_near_one_for_good_predictions(self, reg_data):
        X_val, y_true = reg_data
        # Very low noise → near-perfect predictions
        pipe    = _make_reg_pipeline(y_true, noise=0.001)
        metrics = evaluate_pipeline(pipe, X_val, y_true, task="regression")
        assert metrics["val_r2"] > 0.9

    def test_mae_leq_rmse(self, reg_data):
        """MAE ≤ RMSE always."""
        X_val, y_true = reg_data
        pipe    = _make_reg_pipeline(y_true)
        metrics = evaluate_pipeline(pipe, X_val, y_true, task="regression")
        assert metrics["val_mae"] <= metrics["val_rmse"] + 1e-6


# ── champion_metric ────────────────────────────────────────────

class TestChampionMetric:

    def test_classification_uses_accuracy(self):
        metrics = {"val_accuracy": 0.85, "val_rmse": 0.3}
        assert champion_metric(metrics, "classification") == 0.85

    def test_regression_negates_rmse(self):
        metrics = {"val_rmse": 1234.5, "val_r2": 0.9}
        assert champion_metric(metrics, "regression") == -1234.5

    def test_higher_accuracy_wins(self):
        m1 = {"val_accuracy": 0.90}
        m2 = {"val_accuracy": 0.85}
        assert champion_metric(m1, "classification") > champion_metric(m2, "classification")

    def test_lower_rmse_wins(self):
        m1 = {"val_rmse": 100.0}
        m2 = {"val_rmse": 200.0}
        # Lower RMSE → higher champion_metric (less negative)
        assert champion_metric(m1, "regression") > champion_metric(m2, "regression")
