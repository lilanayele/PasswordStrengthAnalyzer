"""
models/train_lr.py

Trains a Logistic Regression classifier on the 87-dim feature vector.
Saves model + feature extractor to models/saved/lr/

Usage:
    python models/train_lr.py
    python models/train_lr.py --max-rows 100000   # quick test
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.append(str(Path(__file__).parent.parent))
from features.extract_features import FeatureExtractor

SAVE_DIR = Path("models/saved/lr")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


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

    X_val = fe.transform(val["password"].tolist())
    y_val = val["label"].values

    print(f"  Feature matrix: {X_train.shape}")

    print("\n── Training Logistic Regression ──")
    # Class weights for imbalance
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_acc = model.score(X_val, y_val)
    print(f"  Validation accuracy: {val_acc:.4f}")

    # Save
    with open(SAVE_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(SAVE_DIR / "feature_extractor.pkl", "wb") as f:
        pickle.dump(fe, f)

    print(f"\n  Saved to {SAVE_DIR}/")
    print("  ✓ model.pkl")
    print("  ✓ feature_extractor.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    main(args)
