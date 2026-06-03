"""
data/label_passwords.py

Assigns strength labels to passwords using a Monte Carlo Markov-chain approach,
following Dell'Amico & Filippone (2015).

Labels (based on log10 of guessing number G):
  0 = Very Weak  (log G < 6)
  1 = Weak       (6  <= log G < 10)
  2 = Moderate   (10 <= log G < 14)
  3 = Strong     (log G >= 14)
"""

import math
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── Markov chain training ─────────────────────────────────────────────────────

def build_markov_model(passwords: list[str], order: int = 3) -> dict:
    """
    Train an order-n character-level Markov chain on a list of passwords.
    Returns transition probability tables as nested dicts.
    """
    counts = defaultdict(lambda: defaultdict(int))
    start_counts = defaultdict(int)

    for pw in passwords:
        if len(pw) < order + 1:
            continue
        start_counts[pw[:order]] += 1
        for i in range(len(pw) - order):
            context = pw[i:i + order]
            next_char = pw[i + order]
            counts[context][next_char] += 1

    # Convert to log-probabilities
    model = {}
    for context, next_chars in counts.items():
        total = sum(next_chars.values())
        model[context] = {ch: math.log(cnt / total) for ch, cnt in next_chars.items()}

    # Start distribution
    start_total = sum(start_counts.values())
    start_probs = {ctx: math.log(cnt / start_total) for ctx, cnt in start_counts.items()}

    return {"transitions": model, "start": start_probs, "order": order}


def password_log_prob(password: str, model: dict) -> float:
    """
    Compute log-probability of a password under the Markov model.
    Returns -inf if the password contains unseen transitions.
    """
    order = model["order"]
    if len(password) < order:
        return -math.inf

    start_ctx = password[:order]
    if start_ctx not in model["start"]:
        return -math.inf

    log_p = model["start"][start_ctx]
    for i in range(len(password) - order):
        ctx = password[i:i + order]
        nxt = password[i + order]
        if ctx not in model["transitions"] or nxt not in model["transitions"][ctx]:
            return -math.inf
        log_p += model["transitions"][ctx][nxt]

    return log_p


# ── Guessing number estimation ────────────────────────────────────────────────

def estimate_guessing_number(
    password: str,
    model: dict,
    n_samples: int = 10_000,
    rng_seed: int = 42,
) -> float:
    """
    Monte Carlo estimate of the guessing number G for a password.

    G is estimated as: G ≈ 1 / P(password), where P is approximated by
    sampling n_samples passwords from the model and computing the fraction
    with probability >= P(target).

    Returns log10(G). Returns 20.0 (very strong) if the password has zero
    probability under the model (unseen n-gram).
    """
    rng = random.Random(rng_seed)

    target_log_p = password_log_prob(password, model)
    if target_log_p == -math.inf:
        return 20.0  # Unknown structure → treat as strong

    # Sample passwords from the model to estimate rank
    order = model["order"]
    start_pool = list(model["start"].keys())
    start_weights = [math.exp(v) for v in model["start"].values()]

    n_easier = 0
    for _ in range(n_samples):
        # Sample a starting context
        ctx = rng.choices(start_pool, weights=start_weights, k=1)[0]
        sampled = list(ctx)

        # Extend up to 64 chars or until no transitions
        for _step in range(64 - order):
            transitions = model["transitions"].get("".join(sampled[-order:]))
            if not transitions:
                break
            chars = list(transitions.keys())
            weights = [math.exp(v) for v in transitions.values()]
            next_char = rng.choices(chars, weights=weights, k=1)[0]
            sampled.append(next_char)
            if next_char in ("\x00", "\n"):  # End-of-password token
                break

        sample_pw = "".join(sampled)
        sample_log_p = password_log_prob(sample_pw, model)
        if sample_log_p != -math.inf and sample_log_p >= target_log_p:
            n_easier += 1

    if n_easier == 0:
        return 20.0

    estimated_prob = n_easier / n_samples
    return -math.log10(estimated_prob)


# ── Label assignment ──────────────────────────────────────────────────────────

def log_g_to_label(log_g: float) -> int:
    """
    Discretize log10(G) into four ordinal strength classes.
    0=Very Weak, 1=Weak, 2=Moderate, 3=Strong
    """
    if log_g < 6:
        return 0
    elif log_g < 10:
        return 1
    elif log_g < 14:
        return 2
    else:
        return 3


LABEL_NAMES = {0: "Very Weak", 1: "Weak", 2: "Moderate", 3: "Strong"}


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pandas as pd
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Label passwords with Markov MC strength scores")
    parser.add_argument("--input", required=True, help="Path to plaintext password list (one per line)")
    parser.add_argument("--output", required=True, help="Output CSV path (password, log_g, label)")
    parser.add_argument("--model-out", default="data/markov_model.pkl", help="Save trained model")
    parser.add_argument("--train-fraction", type=float, default=0.20,
                        help="Fraction of passwords used to train the Markov model (held-out)")
    parser.add_argument("--order", type=int, default=3, help="Markov chain order (default: 3)")
    parser.add_argument("--samples", type=int, default=5000, help="MC samples per password")
    parser.add_argument("--max-passwords", type=int, default=None,
                        help="Cap total passwords to label (for quick testing)")
    args = parser.parse_args()

    print(f"Loading passwords from {args.input} ...")
    with open(args.input, encoding="latin-1", errors="replace") as f:
        passwords = [line.rstrip("\n") for line in f]

    if args.max_passwords:
        passwords = passwords[: args.max_passwords]

    print(f"  Total passwords: {len(passwords):,}")

    # Train/eval split: use train_fraction as held-out training set for the model
    rng = random.Random(42)
    rng.shuffle(passwords)
    split = int(len(passwords) * args.train_fraction)
    train_pws = passwords[:split]
    eval_pws = passwords[split:]

    print(f"Training Markov model (order={args.order}) on {len(train_pws):,} passwords ...")
    model = build_markov_model(train_pws, order=args.order)

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    print(f"Labeling {len(eval_pws):,} passwords ...")
    rows = []
    for pw in tqdm(eval_pws):
        log_g = estimate_guessing_number(pw, model, n_samples=args.samples)
        label = log_g_to_label(log_g)
        rows.append({"password": pw, "log_g": round(log_g, 3), "label": label})

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    dist = df["label"].value_counts(normalize=True).sort_index()
    print("\nLabel distribution:")
    for lbl, frac in dist.items():
        print(f"  {LABEL_NAMES[lbl]}: {frac*100:.1f}%")
    print(f"\nSaved to {args.output}")
