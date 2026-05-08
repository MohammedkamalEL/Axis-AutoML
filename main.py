"""
Spaceship Titanic — AutoML Pipeline
====================================
Usage:
  python main.py --automl              # full AutoML run (default 50 trials/model)
  python main.py --automl --trials 10  # quick run with fewer trials
  python main.py --eda                 # EDA plots only
  python main.py --all --trials 20    # EDA + AutoML
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


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_df = pd.read_csv("data/train.csv")
    test_df  = pd.read_csv("data/test.csv")
    X = train_df.drop("Transported", axis=1)
    y = train_df["Transported"].astype(int)
    return X, y, test_df


def run_automl(n_trials: int = 50) -> None:
    X, y, test_df = load_data()

    # ── AutoML search ──
    engine = AutoMLEngine(n_trials=n_trials, cv_folds=5, val_size=0.2)
    engine.fit(X, y)

    # ── Leaderboard ──
    print("\n📊 LEADERBOARD")
    print("─" * 70)
    print(engine.leaderboard().to_string())

    # ── Champion report ──
    engine.champion_report()

    # ── Kaggle submission ──
    test_ids    = test_df["PassengerId"].copy()
    predictions = engine.predict(test_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Transported": predictions.astype(bool),
    })
    sub_path = f"submissions/automl_{timestamp}.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\n✅ Submission saved → {sub_path}")
    print(f"   Prediction split: {submission['Transported'].value_counts().to_dict()}")

    # ── Save champion + model card ──
    registry_path = f"model_registry/champion_{timestamp}"
    os.makedirs(registry_path, exist_ok=True)

    joblib.dump(engine.champion,  f"{registry_path}/model.pkl")
    joblib.dump(engine.engineer,  f"{registry_path}/engineer.pkl")

    lb = engine.leaderboard()
    card = {
        "champion":   engine.champion_name,
        "timestamp":  timestamp,
        "n_trials":   n_trials,
        "leaderboard": lb.reset_index().rename(columns={"index": "model"}).to_dict(orient="records"),
        "submission":  sub_path,
    }
    card_path = f"{registry_path}/model_card.json"
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)

    print(f"💾 Champion artifacts saved → {registry_path}/")
    print(
        "\n📈 View MLflow dashboard:\n"
        "   mlflow ui --backend-store-uri sqlite:///spaceship_experiments/mlflow.db"
    )


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spaceship Titanic — AutoML Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--eda",    action="store_true", help="Run EDA plots only")
    parser.add_argument("--automl", action="store_true", help="Run AutoML search")
    parser.add_argument("--all",    action="store_true", help="EDA + AutoML")
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Optuna trials per model (default: 50)"
    )
    args = parser.parse_args()

    if not any([args.eda, args.automl, args.all]):
        parser.print_help()
    else:
        if args.eda or args.all:
            X, y, _ = load_data()
            run_eda(pd.concat([X, y.rename("Transported")], axis=1))

        if args.automl or args.all:
            run_automl(n_trials=args.trials)
