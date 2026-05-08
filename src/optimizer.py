import optuna
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_model(
    model_name: str,
    model_class,
    param_fn,
    X_train,
    y_train,
    preprocessor,
    task: str = "classification",
    n_trials: int = 50,
    cv_folds: int = 5,
) -> tuple:
    """
    Runs Optuna TPE optimization for a single model.

    Parameters
    ----------
    task : "classification" | "regression"
        Controls CV strategy, scoring metric, and Optuna direction.

    Returns
    -------
    (best_pipeline, best_cv_score, best_params)
    """
    # Task-specific settings
    if task == "classification":
        scoring    = "accuracy"
        direction  = "maximize"
        cv         = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        score_sign = 1
    else:
        scoring    = "neg_root_mean_squared_error"
        direction  = "minimize"          # we want lowest RMSE
        cv         = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        score_sign = -1                  # neg_rmse → positive rmse

    def objective(trial):
        params   = param_fn(trial)
        model    = model_class(**params)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=-1,
        )
        return scores.mean() * score_sign   # always pass positive value to Optuna

    study = optuna.create_study(
        direction   = "maximize",           # we always maximise (RMSE already negated above)
        study_name  = f"{model_name}_study",
        sampler     = optuna.samplers.TPESampler(seed=42),
        pruner      = optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params   = param_fn(study.best_trial)
    best_model    = model_class(**best_params)
    best_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", best_model),
    ])
    best_pipeline.fit(X_train, y_train)

    # Return the actual metric value (positive RMSE for regression)
    best_score = study.best_value * score_sign

    return best_pipeline, best_score, best_params
