import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report, confusion_matrix
)


def evaluate_pipeline(pipeline, X_val, y_val) -> dict:
    """Full evaluation of a fitted pipeline on a validation set."""
    y_pred  = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    return {
        "val_accuracy":  round(accuracy_score(y_val, y_pred), 4),
        "val_f1":        round(f1_score(y_val, y_pred), 4),
        "val_roc_auc":   round(roc_auc_score(y_val, y_proba), 4),
        "val_precision": round(precision_score(y_val, y_pred), 4),
        "val_recall":    round(recall_score(y_val, y_pred), 4),
    }


def cv_score(pipeline, X, y, cv_folds: int = 5) -> dict:
    """Quick cross-validation estimate before full tuning."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return {
        "cv_mean": round(scores.mean(), 4),
        "cv_std":  round(scores.std(), 4),
    }


def full_report(pipeline, X_val, y_val) -> None:
    """Prints classification report + confusion matrix."""
    y_pred = pipeline.predict(X_val)
    print("\n📋 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=["Not Transported", "Transported"]))
    print("🔲 Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
