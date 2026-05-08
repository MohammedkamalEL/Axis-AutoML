import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def build_voting_ensemble(trained_pipelines: dict, voting: str = "soft") -> VotingClassifier:
    """
    Builds a soft/hard VotingClassifier from already-fitted pipelines.
    voting='soft' uses probability averaging (usually better for classification).
    """
    estimators = list(trained_pipelines.items())
    return VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)


def build_stacking_ensemble(trained_pipelines: dict, cv: int = 5) -> StackingClassifier:
    """
    Builds a StackingClassifier with LogisticRegression as meta-learner.
    Uses cross-val predictions of base estimators as meta-features.
    """
    estimators = list(trained_pipelines.items())
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)

    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=cv,
        n_jobs=-1,
        passthrough=False,
    )


def auto_weight_ensemble(trained_pipelines: dict, X_val, y_val) -> VotingClassifier:
    """
    Assigns weights proportional to each model's validation accuracy,
    then returns a weighted soft VotingClassifier.
    """
    weights = []
    estimators = []

    for name, pipeline in trained_pipelines.items():
        acc = accuracy_score(y_val, pipeline.predict(X_val))
        weights.append(acc)
        estimators.append((name, pipeline))

    # Scale weights so they sum to len(models) (sklearn convention)
    weights = np.array(weights)
    weights = (weights / weights.sum() * len(weights)).tolist()

    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=weights,
        n_jobs=-1,
    )
