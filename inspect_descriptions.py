import pandas as pd
from pathlib import Path
from collections import Counter
import re

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "uae_used_cars_clean.csv")

descriptions = df["Description"].dropna().str.lower()

# --- Extract feature phrases (everything before "condition:") ---
feature_parts = descriptions.str.extract(r"with (.+?)\.?\s*condition:", expand=False).dropna()

feature_phrases = Counter()
for line in feature_parts:
    for phrase in line.split(","):
        phrase = phrase.strip()
        if phrase:
            feature_phrases[phrase] += 1

print("=" * 55)
print(f"TOP 40 FEATURE PHRASES IN DESCRIPTIONS")
print("(These are the exact strings to match in feature_eng.py)")
print("=" * 55)
for phrase, count in feature_phrases.most_common(40):
    print(f"  {count:5d}x  '{phrase}'")

# --- Extract condition phrases (everything after "condition:") ---
condition_parts = descriptions.str.extract(r"condition:\s*(.+)", expand=False).dropna()

condition_counts = Counter(condition_parts.str.strip())
print("\n" + "=" * 55)
print(f"ALL CONDITION VALUES")
print("=" * 55)
for condition, count in condition_counts.most_common():
    print(f"  {count:5d}x  '{condition}'")
