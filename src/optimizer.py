import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_model(
    model_name: str,
    model_class,
    param_fn,
    X_train,
    y_train,
    preprocessor,
    n_trials: int = 50,
    cv_folds: int = 5,
) -> tuple:
    """
    Runs Optuna TPE optimization for a single model.
    Returns (best_pipeline, best_cv_score, best_params).
    """

    def objective(trial):
        params = param_fn(trial)
        model = model_class(**params)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="accuracy", n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}_study",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = param_fn(study.best_trial)
    best_model = model_class(**best_params)
    best_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", best_model),
    ])
    best_pipeline.fit(X_train, y_train)

    return best_pipeline, study.best_value, best_params
