from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_search_space() -> dict:
    """
    Returns all candidate models + their Optuna search spaces.
    Each value is a callable that takes an Optuna trial and returns params dict.
    """
    return {
        "RandomForest": {
            "model": RandomForestClassifier,
            "params": lambda trial: {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
                "max_depth":         trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier,
            "params": lambda trial: {
                "n_estimators":  trial.suggest_int("n_estimators", 100, 400),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth":     trial.suggest_int("max_depth", 3, 8),
                "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": 42,
            },
        },
        "XGBoost": {
            "model": XGBClassifier,
            "params": lambda trial: {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth":         trial.suggest_int("max_depth", 3, 10),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "eval_metric": "logloss",
                "use_label_encoder": False,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        "LightGBM": {
            "model": LGBMClassifier,
            "params": lambda trial: {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves":       trial.suggest_int("num_leaves", 20, 100),
                "max_depth":        trial.suggest_int("max_depth", -1, 15),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            },
        },
        "LogisticRegression": {
            "model": LogisticRegression,
            "params": lambda trial: {
                "C":      trial.suggest_float("C", 0.001, 100, log=True),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                "max_iter": 1000,
                "random_state": 42,
            },
        },
        "ExtraTrees": {
            "model": ExtraTreesClassifier,
            "params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth":    trial.suggest_int("max_depth", 5, 30),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "random_state": 42,
                "n_jobs": -1,
            },
        },
    }
