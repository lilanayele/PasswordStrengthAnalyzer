"""
models/train_xgboost.py

Trains an XGBoost classifier with Optuna Bayesian hyperparameter search.
Saves model + feature extractor to models/saved/xgboost/

Usage:
    python models/train_xgboost.py
    python models/train_xgboost.py --no-optuna     # skip tuning, use paper defaults
    python models/train_xgboost.py --trials 20     # fewer Optuna trials (faster)
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).parent.parent))
from features.extract_features import FeatureExtractor

SAVE_DIR = Path("models/saved/xgboost")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PAPER_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}


def run_optuna(X_train, y_train, X_val, y_val, n_trials: int = 100):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model.score(X_val, y_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n  Best val accuracy: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def main(args):
    print("── Loading data ──")
    train = pd.read_csv("data/processed/train.csv")
    val   = pd.read_csv("data/processed/val.csv")

    if args.max_rows:
        train = train.sample(args.max_rows, random_state=42)
        val   = val.sample(args.max_rows // 5, random_state=42)

    print(f"  Train: {len(train):,}  |  Val: {len(val):,}")

    print("\n── Extracting features ──")
    fe = FeatureExtractor()
    X_train = fe.fit_transform(train["password"].tolist())
    y_train = train["label"].values
    X_val   = fe.transform(val["password"].tolist())
    y_val   = val["label"].values

    # Class weights
    from collections import Counter
    counts = Counter(y_train)
    n_total = len(y_train)
    class_weights = {k: n_total / (len(counts) * v) for k, v in counts.items()}

    if args.no_optuna:
        print("\n── Training XGBoost with paper default params ──")
        params = PAPER_PARAMS.copy()
    else:
        print(f"\n── Running Optuna ({args.trials} trials) ──")
        best_params = run_optuna(X_train, y_train, X_val, y_val, n_trials=args.trials)
        params = {**PAPER_PARAMS, **best_params}

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    val_acc = model.score(X_val, y_val)
    print(f"\n  Final validation accuracy: {val_acc:.4f}")

    # Save
    model.save_model(str(SAVE_DIR / "model.json"))
    with open(SAVE_DIR / "feature_extractor.pkl", "wb") as f:
        pickle.dump(fe, f)

    print(f"\n  Saved to {SAVE_DIR}/")
    print("  ✓ model.json")
    print("  ✓ feature_extractor.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-optuna", action="store_true")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    main(args)
