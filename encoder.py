import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "uae_cars_merged.csv")

df["log_price"] = np.log1p(df["price"])

# Label encode low-cardinality categoricals
label_encoders = {}
for col in ["body_type", "transmission", "fuel_type", "location"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = {cls: int(i) for i, cls in enumerate(le.classes_)}

# Fill remaining NaN in core numerical columns before anything else
df["cylinders"]     = df["cylinders"].fillna(df["cylinders"].median())
df["engine_size_l"] = df["engine_size_l"].fillna(df["engine_size_l"].median())

# NaN-safe binary features — impute with 0
# "Unknown" treated as "not present" (conservative)
binary_cols = [
    "has_sunroof", "has_leather", "has_camera", "has_cruise",
    "has_navigation", "has_bluetooth", "is_clean_condition",
    "has_accident_history", "needs_repair", "has_repainted_bumper",
    "has_damage",
]
for col in binary_cols:
    df[col] = df[col].fillna(0).astype(int)

# Numerical NaN imputation (engine_size_l already handled above)
df["warranty_months"] = df["warranty_months"].fillna(0)
df["is_gcc_specs"]    = df["is_gcc_specs"].fillna(0).astype(int)

feature_cols = [
    # Core
    "make", "model", "age", "log_mileage", "mileage_per_year", "cylinders",
    "transmission", "fuel_type", "body_type", "location",
    # Cars24-only (NaN-imputed for Kaggle rows)
    "trim", "engine_size_l", "warranty_months", "is_gcc_specs",
    # Description features (0-imputed for Cars24 rows)
    "has_sunroof", "has_leather", "has_camera", "has_cruise",
    "has_navigation", "has_bluetooth",
    # Condition flags
    "is_clean_condition", "has_accident_history", "needs_repair",
    "has_repainted_bumper", "has_damage",
    # Engineered
    "is_neutral_color", "is_high_performance", "is_premium_body",
]

# Split FIRST — before any target-based encoding
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.copy()
test_df  = test_df.copy()

# Rare encoding — replace categories that appear in < 1% of training rows
# with the string "rare" so they share one target-encoded value instead of
# getting individual noisy encodings based on 3-5 data points.
global_mean = train_df["log_price"].mean()

# Target encode make, model, and trim using training data only
target_maps = {}
for col in ["make", "model", "trim"]:
    target_map = train_df.groupby(col)["log_price"].mean()
    target_maps[col] = target_map.to_dict()
    train_df[col] = train_df[col].map(target_map).fillna(global_mean)
    test_df[col]  = test_df[col].map(target_map).fillna(global_mean)

X_train = train_df[feature_cols].values
X_test  = test_df[feature_cols].values
y_train = train_df["log_price"].values
y_test  = test_df["log_price"].values

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Total features: {len(feature_cols)}")

# Print source breakdown in train/test
train_sources = train_df["source"].value_counts()
test_sources  = test_df["source"].value_counts()
print(f"\nTrain — kaggle: {train_sources.get('kaggle',0):,}  cars24: {train_sources.get('cars24',0):,}")
print(f"Test  — kaggle: {test_sources.get('kaggle',0):,}  cars24: {test_sources.get('cars24',0):,}")

np.save(BASE_DIR / "X_train.npy", X_train)
np.save(BASE_DIR / "X_test.npy",  X_test)
np.save(BASE_DIR / "y_train.npy", y_train)
np.save(BASE_DIR / "y_test.npy",  y_test)
print("\nSaved train/test splits.")

encodings = {
    "label_encoders": label_encoders,
    "target_maps": target_maps,
    "global_mean": global_mean,
}
with open(BASE_DIR / "models/encodings.json", "w") as f:
    json.dump(encodings, f)
print("Saved encodings.json")
