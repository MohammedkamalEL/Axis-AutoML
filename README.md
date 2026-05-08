# 🚀 Spaceship Titanic — AutoML Pipeline

A production-ready **AutoML pipeline** that automatically searches, tunes, and ensembles the best models for the Spaceship Titanic Kaggle competition — no manual model selection required.

---

## 🏗 Project Structure

```
spaceship_automl/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── automl_engine.py    ← AutoML core orchestrator
│   ├── search_space.py     ← Model definitions + Optuna param spaces
│   ├── optimizer.py        ← Optuna TPE optimization loop
│   ├── ensemble.py         ← Auto voting / stacking / weighted ensembles
│   ├── evaluator.py        ← CV scoring + validation metrics
│   ├── preprocessing.py    ← SpaceshipFeatureEngineer + ColumnTransformer
│   ├── tracker.py          ← MLflow experiment logging
│   ├── eda.py              ← EDA visualizations
│   └── utils.py            ← Directory setup helpers
├── model_registry/         ← Versioned champion models + model cards
├── submissions/            ← Kaggle CSV submissions
├── spaceship_experiments/  ← MLflow SQLite backend
├── main.py                 ← CLI entry point
└── pyproject.toml
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Add data
Place `train.csv` and `test.csv` inside the `data/` folder.

### 3. Run AutoML
```bash
# Full AutoML run (50 Optuna trials per model)
uv run --no-sync python main.py --automl

# Quick test run (10 trials — faster)
uv run --no-sync python main.py --automl --trials 10

# EDA only
uv run --no-sync python main.py --eda

# EDA + AutoML together
uv run --no-sync python main.py --all --trials 50
```

### 4. View MLflow dashboard
```bash
mlflow ui --backend-store-uri sqlite:///spaceship_experiments/mlflow.db
```

---

## 🤖 How AutoML Works

1. **Feature Engineering** — `SpaceshipFeatureEngineer` extracts group features, decomposes Cabin, creates spending aggregates and age groups.
2. **Search Space** — 6 model families defined with Optuna hyperparameter spaces (RF, GBM, XGBoost, LightGBM, LR, ExtraTrees).
3. **Optuna Optimization** — TPE sampler with MedianPruner, independently optimizes each model family.
4. **Auto Ensembles** — three strategies built automatically:
   - Soft voting
   - Accuracy-weighted voting
   - Stacking with LogisticRegression meta-learner
5. **Champion Selection** — model with highest validation accuracy is selected.
6. **Artifacts** — champion model, feature engineer, and model card saved to `model_registry/`.

---

## 📊 Models Searched

| Model | Algorithm | accurecy
|---|---|---|
| RandomForest | sklearn | 0.8154
| GradientBoosting | sklearn |  0.8045   
| XGBoost | xgboost |  0.8045
| LightGBM | lightgbm |  0.8022
| LogisticRegression | sklearn |  0.793
| ExtraTrees | sklearn |   0.789 
| VotingEnsemble | auto-built | 0.8039 
| WeightedEnsemble | auto-built |  0.8045
| StackingEnsemble | auto-built |  0.8016 

---

## 📈 Results

-🏆 Champion → GradientBoosting  acc=0.8154  roc=0.9105



The champion model and full leaderboard are printed after each run and saved to `model_registry/champion_<timestamp>/model_card.json`.

Current best: ~**80%+ validation accuracy** (improves with more trials).
