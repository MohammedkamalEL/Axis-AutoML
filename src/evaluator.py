import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (
    # Classification
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report, confusion_matrix,
    # Regression
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
)


# ──────────────────────────────────────────────────────────────
# Main evaluation dispatcher
# ──────────────────────────────────────────────────────────────

def evaluate_pipeline(pipeline, X_val, y_val, task: str = "classification") -> dict:
    """
    Evaluates a fitted pipeline on the validation set.

    Parameters
    ----------
    task : "classification" | "regression"

    Returns
    -------
    dict of metric_name → value
    """
    if task == "classification":
        return _eval_classification(pipeline, X_val, y_val)
    else:
        return _eval_regression(pipeline, X_val, y_val)


def cv_score(pipeline, X, y, task: str = "classification", cv_folds: int = 5) -> dict:
    """Quick cross-validation estimate before full tuning."""
    scoring = "accuracy" if task == "classification" else "neg_root_mean_squared_error"

    if task == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    if task == "regression":
        scores = -scores  # neg_rmse → positive rmse

    return {
        "cv_mean": round(scores.mean(), 4),
        "cv_std":  round(scores.std(), 4),
    }


def full_report(pipeline, X_val, y_val, task: str = "classification") -> None:
    """Prints a detailed evaluation report."""
    y_pred = pipeline.predict(X_val)

    if task == "classification":
        print("\n📋 Classification Report:")
        print(classification_report(y_val, y_pred,
                                    target_names=["Class 0", "Class 1"]))
        print("🔲 Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
    else:
        metrics = _eval_regression(pipeline, X_val, y_val)
        print("\n📋 Regression Report:")
        for k, v in metrics.items():
            print(f"  {k:<15} {v}")


# ──────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────

def _eval_classification(pipeline, X_val, y_val) -> dict:
    y_pred  = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    return {
        "val_accuracy":  round(accuracy_score(y_val, y_pred), 4),
        "val_f1":        round(f1_score(y_val, y_pred), 4),
        "val_roc_auc":   round(roc_auc_score(y_val, y_proba), 4),
        "val_precision": round(precision_score(y_val, y_pred), 4),
        "val_recall":    round(recall_score(y_val, y_pred), 4),
    }


def _eval_regression(pipeline, X_val, y_val) -> dict:
    y_pred = pipeline.predict(X_val)
    rmse   = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    return {
        "val_rmse":  round(rmse, 4),
        "val_mae":   round(mean_absolute_error(y_val, y_pred), 4),
        "val_r2":    round(r2_score(y_val, y_pred), 4),
        "val_mape":  round(mean_absolute_percentage_error(y_val, y_pred), 4),
    }


# ──────────────────────────────────────────────────────────────
# Champion metric: higher = better (used to rank models)
# ──────────────────────────────────────────────────────────────

def champion_metric(metrics: dict, task: str) -> float:
    """Returns the single scalar used to pick the champion."""
    if task == "classification":
        return metrics.get("val_accuracy", 0.0)
    else:
        # Lower RMSE is better → negate so max() still works
        return -metrics.get("val_rmse", float("inf"))
