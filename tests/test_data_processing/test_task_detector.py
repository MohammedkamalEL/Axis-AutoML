"""Tests for TaskDetector — all 6 rules + confidence scoring + forced override."""

import pytest
import pandas as pd
import numpy as np

from src.task_detector import TaskDetector, TaskDetectionResult


# ── Helper ─────────────────────────────────────────────────────

def detect(series):
    return TaskDetector().detect(series)


# ── Rule 1: bool dtype ─────────────────────────────────────────

def test_bool_dtype_is_classification(clf_target_bool):
    result = detect(clf_target_bool)
    assert result.task == "classification"


def test_bool_dtype_fires_rule1(clf_target_bool):
    result = detect(clf_target_bool)
    assert any("Rule 1" in r for r in result.rules_fired)


# ── Rule 2: object / string dtype ──────────────────────────────

def test_string_dtype_is_classification(clf_target_str):
    result = detect(clf_target_str)
    assert result.task == "classification"


def test_string_dtype_fires_rule2(clf_target_str):
    result = detect(clf_target_str)
    assert any("Rule 2" in r for r in result.rules_fired)


def test_category_dtype_is_classification():
    y = pd.Series(["yes", "no", "yes"] * 30, dtype="category", name="response")
    assert detect(y).task == "classification"


# ── Rule 3: binary numeric (0/1) ───────────────────────────────

def test_binary_int_is_classification():
    y = pd.Series([0, 1, 1, 0, 1] * 20, name="binary")
    result = detect(y)
    assert result.task == "classification"
    assert any("Rule 3" in r for r in result.rules_fired)


# ── Rule 4: low cardinality ────────────────────────────────────

def test_low_cardinality_int_is_classification(clf_target_int):
    result = detect(clf_target_int)
    assert result.task == "classification"


def test_5_class_target_is_classification():
    y = pd.Series([1, 2, 3, 4, 5] * 40, name="rating")
    assert detect(y).task == "classification"


# ── Rule 5 + 6: regression signals ────────────────────────────

def test_float_continuous_is_regression(reg_target_float):
    result = detect(reg_target_float)
    assert result.task == "regression"


def test_float_target_fires_rule6a(reg_target_float):
    result = detect(reg_target_float)
    assert any("Rule 6a" in r for r in result.rules_fired)


def test_high_cardinality_int_is_regression(reg_target_int_high):
    result = detect(reg_target_int_high)
    assert result.task == "regression"


# ── Confidence ─────────────────────────────────────────────────

def test_confidence_is_between_50_and_100(clf_target_bool, reg_target_float):
    for y in [clf_target_bool, reg_target_float]:
        result = detect(y)
        assert 0.5 <= result.confidence <= 1.0


def test_bool_confidence_is_high(clf_target_bool):
    """Bool is an unambiguous signal — expect >= 75% confidence."""
    result = detect(clf_target_bool)
    assert result.confidence >= 0.75


def test_float_confidence_is_high(reg_target_float):
    """Float continuous is unambiguous — expect >= 75%."""
    result = detect(reg_target_float)
    assert result.confidence >= 0.75


# ── Forced override ────────────────────────────────────────────

def test_forced_classification_overrides_regression(reg_target_float):
    detector = TaskDetector(forced_task="classification")
    result   = detector.detect(reg_target_float)
    assert result.task == "classification"
    assert result.confidence == 1.0


def test_forced_regression_overrides_classification(clf_target_bool):
    detector = TaskDetector(forced_task="regression")
    result   = detector.detect(clf_target_bool)
    assert result.task == "regression"
    assert result.confidence == 1.0


def test_auto_is_same_as_none(clf_target_bool):
    r1 = TaskDetector(forced_task="auto").detect(clf_target_bool)
    r2 = TaskDetector(forced_task=None).detect(clf_target_bool)
    assert r1.task == r2.task


def test_invalid_forced_task_raises():
    with pytest.raises(ValueError):
        TaskDetector(forced_task="unsupervised")


# ── Return type ────────────────────────────────────────────────

def test_returns_task_detection_result(clf_target_bool):
    result = detect(clf_target_bool)
    assert isinstance(result, TaskDetectionResult)


def test_result_has_required_fields(clf_target_bool):
    result = detect(clf_target_bool)
    assert hasattr(result, "task")
    assert hasattr(result, "confidence")
    assert hasattr(result, "rules_fired")
    assert isinstance(result.rules_fired, list)
    assert len(result.rules_fired) > 0


def test_summary_returns_string(clf_target_bool):
    result  = detect(clf_target_bool)
    summary = result.summary()
    assert isinstance(summary, str)
    assert "classification" in summary.lower() or "regression" in summary.lower()
