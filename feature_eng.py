import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "uae_used_cars_clean.csv")

# --- Numerical features ---
df["age"]         = 2025 - df["Year"]
df["log_mileage"] = np.log1p(df["Mileage"])

# clip(lower=1) prevents division by zero for brand-new cars (age=0)
df["mileage_per_year"] = df["Mileage"] / df["age"].clip(lower=1)

# --- Color: neutral colors hold value better in UAE ---
neutral = ["white", "silver", "grey", "gray", "black"]
df["is_neutral_color"] = df["Color"].isin(neutral).astype(int)

# --- High performance engine ---
df["is_high_performance"] = (df["Cylinders"] >= 8).astype(int)

# --- Body type value tier ---
premium_body = ["suv", "coupe", "convertible"]
df["is_premium_body"] = df["Body Type"].str.lower().isin(premium_body).astype(int)

# --- Features from Description ---
# Descriptions use title-case phrases like "Rear camera", "Adaptive cruise control"
# so we match lowercase after .str.lower()
features_to_extract = {
    # Positive features (confirmed present in dataset from inspect_descriptions.py)
    "has_sunroof":    ["sunroof"],
    "has_leather":    ["leather seats"],
    "has_camera":     ["rear camera"],
    "has_cruise":     ["adaptive cruise control"],
    "has_navigation": ["navigation system"],
    "has_bluetooth":  ["bluetooth"],
    # Condition flags — all 6 values confirmed from inspect_descriptions.py
    "is_clean_condition":    ["no damage"],
    "has_accident_history":  ["accident history"],
    "needs_repair":          ["engine repaired"],
    "has_repainted_bumper":  ["repainted bumper"],
    "has_damage":            ["dented door", "minor scratches"],
}

for feature, keywords in features_to_extract.items():
    pattern = "|".join(keywords)
    df[feature] = df["Description"].str.lower().str.contains(
        pattern, na=False
    ).astype(int)

print("Sample descriptions:")
for desc in df["Description"].head(5):
    print(" -", str(desc)[:120])

print("\nFeature hit counts:")
print(df[list(features_to_extract.keys())].sum().sort_values(ascending=False))

df.to_csv(BASE_DIR / "uae_used_cars_features.csv", index=False)
print(f"\nSaved features dataset: {len(df)} rows")
