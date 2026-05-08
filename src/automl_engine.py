import time
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import build_preprocessor, SpaceshipFeatureEngineer
from src.search_space import get_search_space
from src.optimizer import optimize_model
from src.evaluator import evaluate_pipeline, full_report
from src.ensemble import build_voting_ensemble, build_stacking_ensemble, auto_weight_ensemble
from src.tracker import log_run, log_champion


class AutoMLEngine:
    """
    Orchestrates the complete AutoML loop:

    1. Feature engineering  (SpaceshipFeatureEngineer)
    2. For each model in search_space → Optuna TPE optimization
    3. Evaluate all tuned models on hold-out validation set
    4. Auto-build three ensemble strategies:
         • Soft voting
         • Accuracy-weighted voting
         • Stacking with LR meta-learner
    5. Select champion model (highest val_accuracy)
    6. Log every run to MLflow
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        self.n_trials      = n_trials
        self.cv_folds      = cv_folds
        self.val_size      = val_size
        self.random_state  = random_state

        self.results            = {}
        self.trained_pipelines  = {}
        self.champion           = None
        self.champion_name      = None
        self.engineer           = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AutoMLEngine":
        """Run the full AutoML search and return self."""

        # 1. Train / val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y,
        )

        # 2. Feature engineering (fit on train, apply to val)
        self.engineer = SpaceshipFeatureEngineer()
        X_train_eng = self.engineer.fit_transform(X_train)
        X_val_eng   = self.engineer.transform(X_val)

        # 3. Shared preprocessor (ColumnTransformer)
        preprocessor = build_preprocessor()

        search_space = get_search_space()
        n_models     = len(search_space)

        print(
            f"\n🤖 AutoML Search — {n_models} models × {self.n_trials} Optuna trials each\n"
            + "─" * 70
        )

        # 4. Optimize each model
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
                n_trials     = self.n_trials,
                cv_folds     = self.cv_folds,
            )

            val_metrics = evaluate_pipeline(best_pipeline, X_val_eng, y_val)
            elapsed = round(time.time() - t0, 1)

            self.trained_pipelines[model_name] = best_pipeline
            self.results[model_name] = {
                "cv_score": round(best_cv, 4),
                **val_metrics,
                "params": best_params,
                "time_s": elapsed,
            }

            log_run(
                model_name,
                best_params,
                {**val_metrics, "cv_score": round(best_cv, 4)},
                best_pipeline,
            )

            print(
                f"val_acc={val_metrics['val_accuracy']}  "
                f"roc={val_metrics['val_roc_auc']}  "
                f"[{elapsed}s]"
            )

        # 5. Auto ensembles
        print("\n  🔗 Building ensembles...")

        voting = build_voting_ensemble(self.trained_pipelines)
        voting.fit(X_train_eng, y_train)
        self.trained_pipelines["VotingEnsemble"] = voting
        self.results["VotingEnsemble"] = {
            **evaluate_pipeline(voting, X_val_eng, y_val),
            "cv_score": "N/A", "params": {}, "time_s": 0,
        }

        weighted = auto_weight_ensemble(self.trained_pipelines, X_val_eng, y_val)
        weighted.fit(X_train_eng, y_train)
        self.trained_pipelines["WeightedEnsemble"] = weighted
        self.results["WeightedEnsemble"] = {
            **evaluate_pipeline(weighted, X_val_eng, y_val),
            "cv_score": "N/A", "params": {}, "time_s": 0,
        }

        stacking = build_stacking_ensemble(self.trained_pipelines, cv=self.cv_folds)
        stacking.fit(X_train_eng, y_train)
        self.trained_pipelines["StackingEnsemble"] = stacking
        self.results["StackingEnsemble"] = {
            **evaluate_pipeline(stacking, X_val_eng, y_val),
            "cv_score": "N/A", "params": {}, "time_s": 0,
        }

        # 6. Select champion
        self.champion_name = max(
            self.results, key=lambda k: self.results[k]["val_accuracy"]
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
            f"\n{'─'*70}"
            f"\n🏆 Champion → {self.champion_name}"
            f"  acc={self.results[self.champion_name]['val_accuracy']}"
            f"  roc={self.results[self.champion_name].get('val_roc_auc', 'N/A')}"
        )

        return self

    def predict(self, X: pd.DataFrame):
        """Transform + predict using the champion pipeline."""
        if self.engineer is None or self.champion is None:
            raise RuntimeError("Call .fit() before .predict()")
        X_eng = self.engineer.transform(X)
        return self.champion.predict(X_eng)

    def leaderboard(self) -> pd.DataFrame:
        """Returns a DataFrame ranking all models by val_accuracy."""
        rows = {
            name: {
                "val_accuracy":  r.get("val_accuracy"),
                "val_roc_auc":   r.get("val_roc_auc"),
                "val_f1":        r.get("val_f1"),
                "cv_score":      r.get("cv_score"),
                "time_s":        r.get("time_s"),
            }
            for name, r in self.results.items()
        }
        return pd.DataFrame(rows).T.sort_values("val_accuracy", ascending=False)

    def champion_report(self) -> None:
        """Prints full classification report for the champion on the val set."""
        if self.champion is None:
            raise RuntimeError("Call .fit() first.")
        print(f"\n📊 Champion: {self.champion_name}")
        full_report(self.champion, self._X_val, self._y_val)
