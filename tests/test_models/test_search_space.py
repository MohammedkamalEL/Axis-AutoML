"""Tests for search_space — model registry for both tasks."""

import pytest
from unittest.mock import MagicMock

from src.search_space import get_search_space


class TestSearchSpace:

    def test_classification_returns_dict(self):
        space = get_search_space("classification")
        assert isinstance(space, dict)
        assert len(space) > 0

    def test_regression_returns_dict(self):
        space = get_search_space("regression")
        assert isinstance(space, dict)
        assert len(space) > 0

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            get_search_space("clustering")

    def test_each_entry_has_model_key(self):
        for task in ("classification", "regression"):
            for name, spec in get_search_space(task).items():
                assert "model" in spec, f"{name} missing 'model' key"

    def test_each_entry_has_params_callable(self):
        for task in ("classification", "regression"):
            for name, spec in get_search_space(task).items():
                assert callable(spec["params"]), f"{name} 'params' must be callable"

    def test_params_callable_accepts_trial(self):
        """params(trial) must return a dict when called with a mock trial."""
        mock_trial = MagicMock()
        mock_trial.suggest_int.return_value        = 100
        mock_trial.suggest_float.return_value      = 0.1
        mock_trial.suggest_categorical.return_value = "sqrt"

        for task in ("classification", "regression"):
            for name, spec in get_search_space(task).items():
                result = spec["params"](mock_trial)
                assert isinstance(result, dict), f"{name}: params() must return dict"

    def test_classification_has_expected_models(self):
        space = get_search_space("classification")
        expected = {"RandomForest", "XGBoost", "LightGBM", "LogisticRegression"}
        assert expected.issubset(set(space.keys()))

    def test_regression_has_expected_models(self):
        space = get_search_space("regression")
        expected = {"RandomForestRegressor", "XGBoostRegressor", "Ridge", "Lasso"}
        assert expected.issubset(set(space.keys()))

    def test_classification_and_regression_use_different_models(self):
        clf_names = set(get_search_space("classification").keys())
        reg_names = set(get_search_space("regression").keys())
        # They shouldn't be identical
        assert clf_names != reg_names

    def test_model_class_is_instantiatable(self):
        """Verify each model class can be instantiated with default params."""
        mock_trial = MagicMock()
        mock_trial.suggest_int.return_value         = 10
        mock_trial.suggest_float.return_value       = 0.1
        mock_trial.suggest_categorical.return_value = "sqrt"

        for task in ("classification", "regression"):
            for name, spec in get_search_space(task).items():
                params = spec["params"](mock_trial)
                # Should not raise
                try:
                    instance = spec["model"](**params)
                except Exception as e:
                    pytest.fail(f"{name}: could not instantiate with params {params}: {e}")
