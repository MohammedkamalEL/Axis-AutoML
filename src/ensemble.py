import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# ──────────────────────────────────────────────────────────────
# Voting ensembles
# ──────────────────────────────────────────────────────────────

def build_voting_ensemble(trained_pipelines: dict, task: str = "classification"):
    """
    Builds a soft VotingClassifier or a mean VotingRegressor
    from already-fitted pipelines.
    """
    estimators = list(trained_pipelines.items())

    if task == "classification":
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    else:
        return VotingRegressor(estimators=estimators, n_jobs=-1)


def auto_weight_ensemble(
    trained_pipelines: dict,
    X_val,
    y_val,
    task: str = "classification",
):
    """
    Assigns weights to each model based on validation performance,
    then returns a weighted ensemble.

    Classification : weights ∝ accuracy  (higher = better)
    Regression     : weights ∝ 1/RMSE    (lower RMSE = higher weight)
    """
    estimators = []
    weights    = []

    for name, pipeline in trained_pipelines.items():
        y_pred = pipeline.predict(X_val)

        if task == "classification":
            score = accuracy_score(y_val, y_pred)
        else:
            rmse  = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            score = 1.0 / (rmse + 1e-9)   # avoid division by zero

        estimators.append((name, pipeline))
        weights.append(score)

    weights = np.array(weights)
    weights = (weights / weights.sum() * len(weights)).tolist()

    if task == "classification":
        return VotingClassifier(
            estimators=estimators, voting="soft", weights=weights, n_jobs=-1
        )
    else:
        return VotingRegressor(
            estimators=estimators, weights=weights, n_jobs=-1
        )


# ──────────────────────────────────────────────────────────────
# Stacking ensembles
# ──────────────────────────────────────────────────────────────

def build_stacking_ensemble(
    trained_pipelines: dict,
    task: str = "classification",
    cv: int = 5,
):
    """
    Builds a StackingClassifier or StackingRegressor.

    Meta-learner:
        Classification → LogisticRegression
        Regression     → Ridge
    """
    estimators = list(trained_pipelines.items())

    if task == "classification":
        return StackingClassifier(
            estimators      = estimators,
            final_estimator = LogisticRegression(max_iter=1000, random_state=42),
            cv              = cv,
            n_jobs          = -1,
            passthrough     = False,
        )
    else:
        return StackingRegressor(
            estimators      = estimators,
            final_estimator = Ridge(alpha=1.0),
            cv              = cv,
            n_jobs          = -1,
            passthrough     = False,
        )
