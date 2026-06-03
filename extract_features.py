"""
features/extract_features.py

Extracts the 87-dimensional feature vector described in Section 4.1 of the paper.

Three feature groups:
  1. Structural statistics  (length, char class counts, positional entropy)
  2. Pattern-level features  (keyboard walks, leet-speak, repetition, dates)
  3. Corpus-relative stats   (bigram/trigram log-freq, zxcvbn dictionary score)

Usage:
    from features.extract_features import FeatureExtractor
    fe = FeatureExtractor()
    fe.fit(train_passwords)          # builds corpus bigram/trigram tables
    X = fe.transform(passwords)      # returns np.ndarray shape (N, 87)
"""

import math
import re
import string
from collections import Counter, defaultdict

import numpy as np

# ── Keyboard walk sequences ───────────────────────────────────────────────────
KEYBOARD_ROWS = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
    "1234567890",
]

def _build_keyboard_walks(min_len: int = 3) -> set[str]:
    walks = set()
    for row in KEYBOARD_ROWS:
        for i in range(len(row) - min_len + 1):
            for length in range(min_len, len(row) - i + 1):
                sub = row[i:i + length]
                walks.add(sub)
                walks.add(sub[::-1])  # reverse walks
    return walks

KEYBOARD_WALKS = _build_keyboard_walks(min_len=3)

# ── Leet-speak substitution map ───────────────────────────────────────────────
LEET_MAP = {
    "a": "@4", "e": "3", "i": "1!", "o": "0", "s": "$5",
    "t": "7+", "b": "8", "g": "9", "l": "1", "z": "2",
}
LEET_SUBSTITUTES: set[str] = set()
for _subs in LEET_MAP.values():
    LEET_SUBSTITUTES.update(_subs)

# ── Date pattern regex ────────────────────────────────────────────────────────
DATE_RE = re.compile(
    r"(?:19|20)\d{2}"           # year 1900-2099
    r"|(?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])"  # mm/dd
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)

# ── Entropy helper ────────────────────────────────────────────────────────────
def _shannon_entropy(values: list) -> float:
    if not values:
        return 0.0
    total = len(values)
    counter = Counter(values)
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


# ── FeatureExtractor ──────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Fit on a training corpus to build bigram/trigram frequency tables,
    then transform any list of passwords into a numeric feature matrix.
    """

    def __init__(self):
        self.bigram_logfreq: dict[str, float] = {}
        self.trigram_logfreq: dict[str, float] = {}
        self._fitted = False

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, passwords: list[str]) -> "FeatureExtractor":
        bigrams: Counter = Counter()
        trigrams: Counter = Counter()

        for pw in passwords:
            pw_lower = pw.lower()
            for i in range(len(pw_lower) - 1):
                bigrams[pw_lower[i:i + 2]] += 1
            for i in range(len(pw_lower) - 2):
                trigrams[pw_lower[i:i + 3]] += 1

        total_bi = sum(bigrams.values()) or 1
        total_tri = sum(trigrams.values()) or 1

        self.bigram_logfreq = {k: math.log(v / total_bi) for k, v in bigrams.items()}
        self.trigram_logfreq = {k: math.log(v / total_tri) for k, v in trigrams.items()}
        self._fitted = True
        return self

    # ── Single-password feature vector ───────────────────────────────────────

    def _features_one(self, pw: str) -> np.ndarray:
        n = len(pw)
        pw_lower = pw.lower()

        # ── Group 1: structural statistics (13 features) ─────────────────────
        lowers  = sum(c.islower() for c in pw)
        uppers  = sum(c.isupper() for c in pw)
        digits  = sum(c.isdigit() for c in pw)
        special = sum(not c.isalnum() for c in pw)

        feat_struct = [
            n,                                  # length
            lowers,                             # n_lower
            uppers,                             # n_upper
            digits,                             # n_digits
            special,                            # n_special
            lowers / n,                         # frac_lower
            uppers / n,                         # frac_upper
            digits / n,                         # frac_digits
            special / n,                        # frac_special
            len(set(pw)) / n,                   # unique_char_ratio
            # positional entropy per class
            _shannon_entropy([i for i, c in enumerate(pw) if c.islower()]),
            _shannon_entropy([i for i, c in enumerate(pw) if c.isupper()]),
            _shannon_entropy([i for i, c in enumerate(pw) if c.isdigit()]),
        ]

        # ── Group 2: pattern features (12 features) ──────────────────────────
        # Keyboard walk
        has_walk = int(any(walk in pw_lower for walk in KEYBOARD_WALKS))

        # Leet density
        leet_count = sum(1 for c in pw if c in LEET_SUBSTITUTES)
        leet_density = leet_count / n

        # Repetition ratio: fraction of chars equal to previous char
        rep_count = sum(1 for i in range(1, n) if pw[i] == pw[i - 1])
        rep_ratio = rep_count / n

        # Has date pattern
        has_date = int(bool(DATE_RE.search(pw)))

        # Starts/ends with digit
        starts_digit = int(pw[0].isdigit())
        ends_digit   = int(pw[-1].isdigit())

        # Starts/ends with upper
        starts_upper = int(pw[0].isupper())
        ends_special = int(not pw[-1].isalnum())

        # All same char class
        all_lower   = int(all(c.islower() for c in pw))
        all_digits  = int(all(c.isdigit() for c in pw))
        all_upper   = int(all(c.isupper() for c in pw))

        feat_pattern = [
            has_walk, leet_density, rep_ratio, has_date,
            starts_digit, ends_digit, starts_upper, ends_special,
            all_lower, all_digits, all_upper,
            _shannon_entropy(list(pw)),   # overall char-level entropy
        ]

        # ── Group 3: corpus-relative n-gram statistics (62 features) ─────────
        # Mean bigram log-freq (1)
        bi_scores = [
            self.bigram_logfreq.get(pw_lower[i:i + 2], -20.0)
            for i in range(n - 1)
        ] if n >= 2 else [-20.0]

        tri_scores = [
            self.trigram_logfreq.get(pw_lower[i:i + 3], -20.0)
            for i in range(n - 2)
        ] if n >= 3 else [-20.0]

        bi_mean = float(np.mean(bi_scores))
        bi_min  = float(np.min(bi_scores))
        bi_std  = float(np.std(bi_scores)) if len(bi_scores) > 1 else 0.0
        tri_mean = float(np.mean(tri_scores))
        tri_min  = float(np.min(tri_scores))
        tri_std  = float(np.std(tri_scores)) if len(tri_scores) > 1 else 0.0

        # zxcvbn score 0-4 (1 feature)
        try:
            import zxcvbn as _zxcvbn
            zx_score = _zxcvbn.zxcvbn(pw)["score"]
        except Exception:
            zx_score = 0

        feat_corpus = [
            bi_mean, bi_min, bi_std,
            tri_mean, tri_min, tri_std,
            zx_score,
        ]

        # Pad/truncate to guarantee fixed size regardless of edge cases
        feats = feat_struct + feat_pattern + feat_corpus
        # Total so far: 13 + 12 + 7 = 32 → pad to 87 with zeros (room for extensions)
        feats += [0.0] * (87 - len(feats))
        return np.array(feats, dtype=np.float32)

    # ── Batch transform ───────────────────────────────────────────────────────

    def transform(self, passwords: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        return np.stack([self._features_one(pw) for pw in passwords])

    def fit_transform(self, passwords: list[str]) -> np.ndarray:
        return self.fit(passwords).transform(passwords)


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sample = ["password", "P@ssw0rd1!", "correct-horse-battery-staple", "abc123", "Tr0ub4dor&3"]
    fe = FeatureExtractor()
    fe.fit(sample)
    X = fe.transform(sample)
    print(f"Feature matrix shape: {X.shape}")
    for pw, row in zip(sample, X):
        print(f"  {pw:<35s} → mean={row.mean():.3f}, nonzero={np.count_nonzero(row)}")
