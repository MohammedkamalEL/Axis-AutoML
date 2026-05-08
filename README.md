# 🚀 The end-to-end AutoML engine: Drop your data, it detects and solves both Classification & Regression tasks

A production-ready **AutoML pipeline** that automatically detects the task type (classification or regression), searches, tunes, and ensembles the best models — no manual configuration required.

---

## 🏗 Project Structure

```
spaceship_automl/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── automl_engine.py    ← AutoML core orchestrator (auto task detection)
│   ├── search_space.py     ← Model definitions + Optuna param spaces (classification & regression)
│   ├── optimizer.py        ← Optuna TPE optimization loop (scoring adapts to task)
│   ├── ensemble.py         ← Auto voting / stacking / weighted ensembles
│   ├── evaluator.py        ← CV scoring + validation metrics (accuracy / RMSE / R²)
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
# Full AutoML run — task auto-detected from target column (default: 50 trials/model)
uv run --no-sync python main.py --automl

# Quick test run (10 trials — faster)
uv run --no-sync python main.py --automl --trials 10

# Explicit target column (default: last column in train.csv)
uv run --no-sync python main.py --automl --target Transported

# Force task type manually
uv run --no-sync python main.py --automl --task regression --target SalePrice

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

### Task Auto-Detection
The engine inspects the target column at runtime and decides the task automatically — no manual configuration needed:

| Condition | Task |
|---|---|
| dtype is `bool` or `object` / `category` | Classification |
| Unique values ≤ 20 | Classification |
| Otherwise (continuous numeric) | Regression |

You can always override with `--task classification` or `--task regression`.

### Pipeline Steps
1. **Feature Engineering** — `SpaceshipFeatureEngineer` extracts group features, decomposes Cabin, creates spending aggregates and age groups.
2. **Search Space** — model families with Optuna hyperparameter spaces, selected per task type.
3. **Optuna Optimization** — TPE sampler with MedianPruner; scoring metric switches automatically (`accuracy` for classification, `neg_RMSE` for regression).
4. **Auto Ensembles** — three strategies built automatically:
   - Soft / mean voting
   - Performance-weighted voting (accuracy weight for classification, 1/RMSE weight for regression)
   - Stacking (LogisticRegression meta-learner for classification, Ridge for regression)
5. **Champion Selection** — highest `val_accuracy` for classification; lowest `val_RMSE` for regression.
6. **Artifacts** — champion model, feature engineer, and model card saved to `model_registry/`.

---

## 📊 Models Searched

### Classification (6 base models + 3 auto ensembles)

| Model | Library | val_accuracy |
|---|---|---|
| 🏆 RandomForest | sklearn | 0.8154 |
| GradientBoosting | sklearn | 0.8045 |
| XGBoost | xgboost | 0.8045 |
| WeightedEnsemble | auto-built | 0.8045 |
| VotingEnsemble | auto-built | 0.8039 |
| StackingEnsemble | auto-built | 0.8016 |
| LightGBM | lightgbm | 0.8022 |
| LogisticRegression | sklearn | 0.7930 |
| ExtraTrees | sklearn | 0.7890 |

### Regression (8 base models + 3 auto ensembles)

| Model | Library | Metric |
|---|---|---|
| RandomForestRegressor | sklearn | val_rmse / val_r2 |
| GradientBoostingRegressor | sklearn | val_rmse / val_r2 |
| XGBoostRegressor | xgboost | val_rmse / val_r2 |
| LightGBMRegressor | lightgbm | val_rmse / val_r2 |
| Ridge | sklearn | val_rmse / val_r2 |
| Lasso | sklearn | val_rmse / val_r2 |
| ElasticNet | sklearn | val_rmse / val_r2 |
| ExtraTreesRegressor | sklearn | val_rmse / val_r2 |
| VotingEnsemble | auto-built | val_rmse / val_r2 |
| WeightedEnsemble | auto-built | val_rmse / val_r2 |
| StackingEnsemble | auto-built | val_rmse / val_r2 |

---

## 📈 Results

### Spaceship Titanic (Classification)
> 🏆 Champion → **RandomForest** — `val_accuracy=0.8154` · `val_roc_auc=0.9105`

The champion model and full leaderboard are printed after each run and saved to `model_registry/champion_<timestamp>/model_card.json`.

Current best: **81.5% validation accuracy** (improves with more trials).
