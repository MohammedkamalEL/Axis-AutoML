"""
Spaceship Titanic — AutoML Pipeline
=====================================
Usage
-----
  python main.py --automl                          # auto-detect task, 50 trials
  python main.py --automl --target SalePrice       # explicit target column
  python main.py --automl --task regression        # force task type
  python main.py --automl --trials 10              # quick run
  python main.py --eda                             # EDA plots only
  python main.py --all --trials 20                 # EDA + AutoML
"""

import argparse
import json
import os
import pandas as pd
import joblib
from datetime import datetime

from src.automl_engine import AutoMLEngine
from src.eda import run_eda
from src.utils import setup_dirs

DIRS = setup_dirs()

DEFAULT_TRAIN = "data/train.csv"
DEFAULT_TEST  = "data/test.csv"


# ──────────────────────────────────────────────────────────────
# Data loading (generic)
# ──────────────────────────────────────────────────────────────

def load_data(
    target_col: str | None = None,
    train_path: str = DEFAULT_TRAIN,
    test_path:  str = DEFAULT_TEST,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Loads train/test CSVs.

    Target column detection order:
    1. Explicit --target argument
    2. Last column of train.csv  (most datasets follow this convention)
    """
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    if target_col is None:
        target_col = train_df.columns[-1]
        print(f"ℹ️  No --target specified. Using last column: '{target_col}'")

    if target_col not in train_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in {train_path}.\n"
            f"Available columns: {list(train_df.columns)}"
        )

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    return X, y, test_df


# ──────────────────────────────────────────────────────────────
# Main AutoML run
# ──────────────────────────────────────────────────────────────

def run_automl(
    n_trials:    int = 50,
    target_col:  str | None = None,
    task:        str = "auto",
    train_path:  str = DEFAULT_TRAIN,
    test_path:   str = DEFAULT_TEST,
) -> None:

    X, y, test_df = load_data(target_col, train_path, test_path)

    engine = AutoMLEngine(
        task         = task,
        n_trials     = n_trials,
        cv_folds     = 5,
        val_size     = 0.2,
    )
    engine.fit(X, y)

    # ── Leaderboard ──────────────────────────────────────────
    print("\n📊 LEADERBOARD")
    print("─" * 72)
    print(engine.leaderboard().to_string())

    engine.champion_report()

    # ── Submission / predictions ──────────────────────────────
    test_ids    = _get_id_column(test_df)
    predictions = engine.predict(test_df)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_name = y.name

    submission = pd.DataFrame({
        "Id":         test_ids,
        target_name:  _format_predictions(predictions, engine._task),
    })
    sub_path = f"submissions/automl_{timestamp}.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\n✅ Submission saved → {sub_path}")

    # ── Save champion artifacts ───────────────────────────────
    registry_path = f"model_registry/champion_{timestamp}"
    os.makedirs(registry_path, exist_ok=True)

    joblib.dump(engine.champion,  f"{registry_path}/model.pkl")
    joblib.dump(engine.engineer,  f"{registry_path}/engineer.pkl")

    card = {
        "champion":    engine.champion_name,
        "task":        engine._task,
        "target_col":  target_name,
        "timestamp":   timestamp,
        "n_trials":    n_trials,
        "leaderboard": engine.leaderboard()
                             .reset_index()
                             .rename(columns={"index": "model"})
                             .to_dict(orient="records"),
        "submission":  sub_path,
    }
    card_path = f"{registry_path}/model_card.json"
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)

    print(f"💾 Champion artifacts → {registry_path}/")
    print(
        "\n📈 MLflow dashboard:\n"
        "   mlflow ui --backend-store-uri sqlite:///spaceship_experiments/mlflow.db"
    )


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _get_id_column(test_df: pd.DataFrame) -> pd.Series:
    """Returns the first column that looks like an ID, else row index."""
    id_candidates = [c for c in test_df.columns
                     if c.lower() in ("id", "passengerid", "customerid", "row_id")]
    if id_candidates:
        return test_df[id_candidates[0]]
    return pd.RangeIndex(len(test_df))


def _format_predictions(predictions, task: str):
    """For classification, convert int 0/1 back to bool if appropriate."""
    if task == "classification":
        return predictions.astype(bool)
    return predictions


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spaceship Titanic — AutoML Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--eda",    action="store_true",
                        help="Run EDA plots only")
    parser.add_argument("--automl", action="store_true",
                        help="Run AutoML search")
    parser.add_argument("--all",    action="store_true",
                        help="EDA + AutoML")

    parser.add_argument("--target", type=str, default=None,
                        help="Target column name (default: last column in train.csv)")
    parser.add_argument("--task",   type=str, default="auto",
                        choices=["auto", "classification", "regression"],
                        help="Force task type (default: auto-detect)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Optuna trials per model (default: 50)")
    parser.add_argument("--train",  type=str, default=DEFAULT_TRAIN,
                        help=f"Path to training CSV (default: {DEFAULT_TRAIN})")
    parser.add_argument("--test",   type=str, default=DEFAULT_TEST,
                        help=f"Path to test CSV (default: {DEFAULT_TEST})")

    args = parser.parse_args()

    if not any([args.eda, args.automl, args.all]):
        parser.print_help()
    else:
        if args.eda or args.all:
            X, y, _ = load_data(args.target, args.train, args.test)
            run_eda(pd.concat([X, y], axis=1))

        if args.automl or args.all:
            run_automl(
                n_trials   = args.trials,
                target_col = args.target,
                task       = args.task,
                train_path = args.train,
                test_path  = args.test,
            )
