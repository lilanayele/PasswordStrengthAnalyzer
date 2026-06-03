"""
data/preprocess.py

Cleans, filters, deduplicates, labels, and splits the raw password datasets.

Usage:
    python data/preprocess.py
    python data/preprocess.py --rockyou-only          # skip LinkedIn
    python data/preprocess.py --max-rows 500000       # quick test run
"""

import argparse
import os
import pickle
import random
import re
import string
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from label_passwords import build_markov_model, estimate_guessing_number, log_g_to_label

# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

ROCKYOU_PATH = RAW_DIR / "rockyou.txt"
LINKEDIN_PATH = RAW_DIR / "linkedin.txt"
MODEL_PATH = Path(__file__).parent / "markov_model.pkl"

MIN_LEN = 4
MAX_LEN = 64
PRINTABLE = set(string.printable)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = 0.15 (remainder)

MC_SAMPLES = 5_000   # increase to 10_000 for paper-quality labels (slower)
MARKOV_ORDER = 3
MARKOV_TRAIN_FRAC = 0.20  # fraction of corpus used to train the Markov model


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_valid(pw: str) -> bool:
    return (
        MIN_LEN <= len(pw) <= MAX_LEN
        and all(c in PRINTABLE for c in pw)
    )


def load_passwords(path: Path, max_rows: int | None = None) -> list[str]:
    passwords = []
    with open(path, encoding="latin-1", errors="replace") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            pw = line.rstrip("\n")
            if is_valid(pw):
                passwords.append(pw)
    return passwords


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    print("── Loading raw passwords ──")
    passwords = load_passwords(ROCKYOU_PATH, max_rows=args.max_rows)
    print(f"  RockYou:  {len(passwords):>10,}")

    if not args.rockyou_only and LINKEDIN_PATH.exists():
        linkedin = load_passwords(LINKEDIN_PATH, max_rows=args.max_rows)
        print(f"  LinkedIn: {len(linkedin):>10,}")
        passwords.extend(linkedin)
    elif not args.rockyou_only:
        print("  LinkedIn: not found — using RockYou only")

    # Deduplicate (keep frequency count as a feature)
    from collections import Counter
    freq = Counter(passwords)
    passwords = list(freq.keys())
    print(f"  After dedup: {len(passwords):,}")

    # Shuffle
    rng = random.Random(42)
    rng.shuffle(passwords)

    # ── Train Markov model on held-out 20% ───────────────────────────────────
    split = int(len(passwords) * MARKOV_TRAIN_FRAC)
    markov_train = passwords[:split]
    to_label = passwords[split:]

    print(f"\n── Training Markov model on {len(markov_train):,} passwords ──")
    model = build_markov_model(markov_train, order=MARKOV_ORDER)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved to {MODEL_PATH}")

    # ── Label passwords ──────────────────────────────────────────────────────
    print(f"\n── Labeling {len(to_label):,} passwords (MC samples={MC_SAMPLES}) ──")
    print("  This may take a while. Use --max-rows 100000 for a quick test.")

    rows = []
    for pw in tqdm(to_label):
        log_g = estimate_guessing_number(pw, model, n_samples=MC_SAMPLES)
        label = log_g_to_label(log_g)
        rows.append({
            "password": pw,
            "frequency": freq.get(pw, 1),
            "log_g": round(log_g, 3),
            "label": label,
        })

    df = pd.DataFrame(rows)

    # ── Distribution report ──────────────────────────────────────────────────
    label_names = {0: "Very Weak", 1: "Weak", 2: "Moderate", 3: "Strong"}
    print("\nLabel distribution:")
    for lbl in sorted(label_names):
        n = (df["label"] == lbl).sum()
        print(f"  {label_names[lbl]:12s}: {n:>8,}  ({n/len(df)*100:.1f}%)")

    # ── Stratified train/val/test split ──────────────────────────────────────
    train_df, temp_df = train_test_split(
        df, test_size=(1 - TRAIN_FRAC), stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"\n── Saved splits ──")
    print(f"  Train: {len(train_df):,} rows → data/processed/train.csv")
    print(f"  Val:   {len(val_df):,} rows → data/processed/val.csv")
    print(f"  Test:  {len(test_df):,} rows → data/processed/test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rockyou-only", action="store_true",
                        help="Use RockYou dataset only (skip LinkedIn)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit total rows per source file (for quick testing)")
    args = parser.parse_args()
    main(args)
