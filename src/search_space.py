from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, RandomForestRegressor,
    GradientBoostingRegressor, ExtraTreesRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


def get_search_space(task: str = "classification") -> dict:
    """
    Returns candidate models + Optuna search spaces.

    Parameters
    ----------
    task : "classification" | "regression"
    """
    if task == "classification":
        return _classification_space()
    elif task == "regression":
        return _regression_space()
    else:
        raise ValueError(f"Unknown task '{task}'. Use 'classification' or 'regression'.")


# ── Classification ─────────────────────────────────────────────

def _classification_space() -> dict:
    return {
        "RandomForest": {
            "model": RandomForestClassifier,
            "params": lambda trial: {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
                "max_depth":         trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "random_state": 42, "n_jobs": -1,
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
                "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth":        trial.suggest_int("max_depth", 3, 10),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "eval_metric": "logloss", "use_label_encoder": False,
                "random_state": 42, "n_jobs": -1,
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
                "random_state": 42, "n_jobs": -1, "verbose": -1,
            },
        },
        "LogisticRegression": {
            "model": LogisticRegression,
            "params": lambda trial: {
                "C":      trial.suggest_float("C", 0.001, 100, log=True),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                "max_iter": 1000, "random_state": 42,
            },
        },
        "ExtraTrees": {
            "model": ExtraTreesClassifier,
            "params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth":    trial.suggest_int("max_depth", 5, 30),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "random_state": 42, "n_jobs": -1,
            },
        },
    }


# ── Regression ─────────────────────────────────────────────────

def _regression_space() -> dict:
    return {
        "RandomForestRegressor": {
            "model": RandomForestRegressor,
            "params": lambda trial: {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
                "max_depth":         trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "random_state": 42, "n_jobs": -1,
            },
        },
        "GradientBoostingRegressor": {
            "model": GradientBoostingRegressor,
            "params": lambda trial: {
                "n_estimators":  trial.suggest_int("n_estimators", 100, 400),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth":     trial.suggest_int("max_depth", 3, 8),
                "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": 42,
            },
        },
        "XGBoostRegressor": {
            "model": XGBRegressor,
            "params": lambda trial: {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth":        trial.suggest_int("max_depth", 3, 10),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": 42, "n_jobs": -1,
            },
        },
        "LightGBMRegressor": {
            "model": LGBMRegressor,
            "params": lambda trial: {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves":       trial.suggest_int("num_leaves", 20, 100),
                "max_depth":        trial.suggest_int("max_depth", -1, 15),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": 42, "n_jobs": -1, "verbose": -1,
            },
        },
        "Ridge": {
            "model": Ridge,
            "params": lambda trial: {
                "alpha": trial.suggest_float("alpha", 0.001, 100, log=True),
            },
        },
        "Lasso": {
            "model": Lasso,
            "params": lambda trial: {
                "alpha": trial.suggest_float("alpha", 0.001, 10, log=True),
                "max_iter": 2000,
            },
        },
        "ElasticNet": {
            "model": ElasticNet,
            "params": lambda trial: {
                "alpha":    trial.suggest_float("alpha", 0.001, 10, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "max_iter": 2000,
            },
        },
        "ExtraTreesRegressor": {
            "model": ExtraTreesRegressor,
            "params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth":    trial.suggest_int("max_depth", 5, 30),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "random_state": 42, "n_jobs": -1,
            },
        },
    }
