"""
AutoMLEngine
============
Detects whether the target is classification or regression automatically,
then runs the complete search → tune → ensemble → champion loop.

Auto-detection rules
--------------------
1. dtype is bool or object/category          → classification
2. nunique ≤ CLASSIFICATION_THRESHOLD        → classification
3. Otherwise                                 → regression

You can always override by passing task="classification" | "regression".
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocessing import build_preprocessor, SpaceshipFeatureEngineer
from src.search_space import get_search_space
from src.optimizer import optimize_model
from src.evaluator import evaluate_pipeline, full_report, champion_metric
from src.ensemble import build_voting_ensemble, build_stacking_ensemble, auto_weight_ensemble
from src.tracker import log_run, log_champion
from src.task_detector import TaskDetector


class AutoMLEngine:
    """
    Fully automated ML pipeline that works for both classification and regression.

    Parameters
    ----------
    task : "auto" | "classification" | "regression"
        "auto" (default) — inferred from the target column at fit time.
    n_trials  : Optuna trials per model   (default 50)
    cv_folds  : cross-validation folds    (default 5)
    val_size  : hold-out fraction         (default 0.2)
    """

    def __init__(
        self,
        task: str = "auto",
        n_trials: int = 50,
        cv_folds: int = 5,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        self.task         = task
        self.n_trials     = n_trials
        self.cv_folds     = cv_folds
        self.val_size     = val_size
        self.random_state = random_state

        # set after fit
        self._task: str | None          = None
        self.engineer                   = None
        self.results: dict              = {}
        self.trained_pipelines: dict    = {}
        self.champion                   = None
        self.champion_name: str | None  = None
        self._X_val                     = None
        self._y_val                     = None

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AutoMLEngine":
        """Detect task, search, tune, ensemble, and select champion."""

        # 1. Detect / validate task
        detector   = TaskDetector(forced_task=self.task)
        detection  = detector.detect(y)
        self._task = detection.task

        print(f"\n🔍 Task Detection")
        print(detection.summary())
        print(f"   Target: '{y.name}'  |  dtype: {y.dtype}  |  unique values: {y.nunique()}")

        # 2. Encode bool target for classification
        if self._task == "classification":
            y = y.astype(int)

        # 3. Train / val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size    = self.val_size,
            random_state = self.random_state,
            stratify     = y if self._task == "classification" else None,
        )

        # 4. Feature engineering
        self.engineer = SpaceshipFeatureEngineer()
        X_train_eng   = self.engineer.fit_transform(X_train)
        X_val_eng     = self.engineer.transform(X_val)

        # 5. Shared preprocessor (ColumnTransformer)
        preprocessor = build_preprocessor()

        # 6. Search space for the detected task
        search_space = get_search_space(task=self._task)
        n_models     = len(search_space)

        print(
            f"\n🤖 AutoML Search — {n_models} models × {self.n_trials} Optuna trials each\n"
            + "─" * 72
        )

        # 7. Optimize each model
        for idx, (model_name, spec) in enumerate(search_space.items(), 1):
            print(f"  [{idx}/{n_models}] Tuning {model_name}...", end=" ", flush=True)
            t0 = time.time()

            best_pipeline, best_cv, best_params = optimize_model(
                model_name   = model_name,
                model_class  = spec["model"],
                param_fn     = spec["params"],
                X_train      = X_train_eng,
                y_train      = y_train,
                preprocessor = preprocessor,
                task         = self._task,
                n_trials     = self.n_trials,
                cv_folds     = self.cv_folds,
            )

            val_metrics = evaluate_pipeline(best_pipeline, X_val_eng, y_val, task=self._task)
            elapsed     = round(time.time() - t0, 1)

            self.trained_pipelines[model_name] = best_pipeline
            self.results[model_name] = {
                "cv_score": round(best_cv, 4),
                **val_metrics,
                "params": best_params,
                "time_s": elapsed,
            }

            log_run(
                model_name, best_params,
                {**val_metrics, "cv_score": round(best_cv, 4)},
                best_pipeline,
            )

            # Print primary metric
            primary = self._primary_metric_str(val_metrics)
            print(f"{primary}  [{elapsed}s]")

        # 8. Auto ensembles
        print("\n  🔗 Building ensembles...")
        self._build_ensembles(X_train_eng, y_train, X_val_eng, y_val)

        # 9. Select champion (highest champion_metric score)
        self.champion_name = max(
            self.results,
            key=lambda k: champion_metric(self.results[k], self._task),
        )
        self.champion  = self.trained_pipelines[self.champion_name]
        self._X_val    = X_val_eng
        self._y_val    = y_val

        champion_metrics = {
            k: v for k, v in self.results[self.champion_name].items()
            if isinstance(v, (int, float))
        }
        log_champion(self.champion_name, champion_metrics, self.champion)

        print(
            f"\n{'─'*72}"
            f"\n🏆 Champion → {self.champion_name}"
        )
        for k, v in self.results[self.champion_name].items():
            if isinstance(v, float):
                print(f"   {k}: {v}")

        return self

    def predict(self, X: pd.DataFrame):
        """Apply feature engineering + predict using the champion."""
        if self.champion is None:
            raise RuntimeError("Call .fit() before .predict()")
        X_eng = self.engineer.transform(X)
        return self.champion.predict(X_eng)

    def leaderboard(self) -> pd.DataFrame:
        """DataFrame of all models ranked by their primary validation metric."""
        primary_key = "val_accuracy" if self._task == "classification" else "val_rmse"
        ascending   = self._task == "regression"   # lower RMSE is better

        rows = {
            name: {
                "cv_score": r.get("cv_score"),
                **{k: v for k, v in r.items()
                   if k.startswith("val_") and isinstance(v, float)},
                "time_s": r.get("time_s"),
            }
            for name, r in self.results.items()
        }
        df = pd.DataFrame(rows).T
        if primary_key in df.columns:
            df = df.sort_values(primary_key, ascending=ascending)
        return df

    def champion_report(self) -> None:
        """Full classification report or regression summary for the champion."""
        if self.champion is None:
            raise RuntimeError("Call .fit() first.")
        print(f"\n📊 Champion: {self.champion_name}  (task={self._task})")
        full_report(self.champion, self._X_val, self._y_val, task=self._task)

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _primary_metric_str(self, metrics: dict) -> str:
        if self._task == "classification":
            return (
                f"val_acc={metrics.get('val_accuracy')}  "
                f"roc={metrics.get('val_roc_auc')}"
            )
        else:
            return (
                f"val_rmse={metrics.get('val_rmse')}  "
                f"val_r2={metrics.get('val_r2')}"
            )

    def _build_ensembles(self, X_train, y_train, X_val, y_val) -> None:
        """Builds and evaluates the three auto ensemble strategies."""
        _no_meta = {"cv_score": "N/A", "params": {}, "time_s": 0}

        # Soft/mean voting
        voting = build_voting_ensemble(self.trained_pipelines, task=self._task)
        voting.fit(X_train, y_train)
        self.trained_pipelines["VotingEnsemble"] = voting
        self.results["VotingEnsemble"] = {
            **evaluate_pipeline(voting, X_val, y_val, task=self._task), **_no_meta
        }

        # Accuracy/RMSE-weighted voting
        weighted = auto_weight_ensemble(
            self.trained_pipelines, X_val, y_val, task=self._task
        )
        weighted.fit(X_train, y_train)
        self.trained_pipelines["WeightedEnsemble"] = weighted
        self.results["WeightedEnsemble"] = {
            **evaluate_pipeline(weighted, X_val, y_val, task=self._task), **_no_meta
        }

        # Stacking
        stacking = build_stacking_ensemble(
            self.trained_pipelines, task=self._task, cv=self.cv_folds
        )
        stacking.fit(X_train, y_train)
        self.trained_pipelines["StackingEnsemble"] = stacking
        self.results["StackingEnsemble"] = {
            **evaluate_pipeline(stacking, X_val, y_val, task=self._task), **_no_meta
        }

        # Print ensemble results
        for name in ("VotingEnsemble", "WeightedEnsemble", "StackingEnsemble"):
            m   = self.results[name]
            out = self._primary_metric_str(m)
            print(f"    ✔ {name}: {out}")
