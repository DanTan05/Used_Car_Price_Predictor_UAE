"""
Merges the Kaggle dataset (9k rows, description features) with the
Cars24 dataset (641 rows, trim/engine/accident features) into a single
unified CSV for training.

Pipeline position: runs AFTER feature_eng.py
Input:  uae_used_cars_features.csv  (Kaggle, already cleaned + engineered)
        cars24_raw.csv               (Cars24 scrape)
Output: uae_cars_merged.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Load Kaggle data ───────────────────────────────────────────────────────
kaggle = pd.read_csv(BASE_DIR / "uae_used_cars_features.csv")

# Apply same price cap as before
kaggle = kaggle[(kaggle["Price"] >= 5_000) & (kaggle["Price"] <= 300_000)].copy()

kaggle_mapped = pd.DataFrame({
    "source":             "kaggle",
    "make":               kaggle["Make"],
    "model":              kaggle["Model"],
    "year":               kaggle["Year"],
    "price":              kaggle["Price"],
    "mileage":            kaggle["Mileage"],
    "body_type":          kaggle["Body Type"].str.lower().str.strip(),
    "cylinders":          kaggle["Cylinders"],
    "transmission":       kaggle["Transmission"].str.lower().str.strip(),
    "fuel_type":          kaggle["Fuel Type"].str.lower().str.strip(),
    "color":              kaggle["Color"],
    "location":           kaggle["Location"].str.lower().str.strip(),
    # Pre-engineered numerical features
    "age":                kaggle["age"],
    "log_mileage":        kaggle["log_mileage"],
    "mileage_per_year":   kaggle["mileage_per_year"],
    # Pre-engineered binary features
    "is_neutral_color":   kaggle["is_neutral_color"],
    "is_high_performance":kaggle["is_high_performance"],
    "is_premium_body":    kaggle["is_premium_body"],
    # Description-extracted features (0/1 already)
    "has_sunroof":        kaggle["has_sunroof"],
    "has_leather":        kaggle["has_leather"],
    "has_camera":         kaggle["has_camera"],
    "has_cruise":         kaggle["has_cruise"],
    "has_navigation":     kaggle["has_navigation"],
    "has_bluetooth":      kaggle["has_bluetooth"],
    "is_clean_condition": kaggle["is_clean_condition"],
    "has_accident_history":kaggle["has_accident_history"],
    "needs_repair":       kaggle["needs_repair"],
    "has_repainted_bumper":kaggle["has_repainted_bumper"],
    "has_damage":         kaggle["has_damage"],
    # Cars24-only columns — NaN for all Kaggle rows
    "trim":               np.nan,
    "engine_size_l":      np.nan,
    "warranty_months":    np.nan,
    "is_gcc_specs":       np.nan,
})

# ── Load Cars24 data ───────────────────────────────────────────────────────
c24 = pd.read_csv(BASE_DIR / "cars24_raw.csv")
c24 = c24[(c24["price_aed"] >= 5_000) & (c24["price_aed"] <= 300_000)].copy()

# Derive condition flags from the structured accident_free column
acc = c24["accident_free"].str.lower().fillna("")
c24_is_clean          = acc.isin(["no accident", "accident free"]).astype(int)
c24_has_accident      = acc.isin(["minor accident", "major accident"]).astype(int)

c24_age = 2025 - pd.to_numeric(c24["year"], errors="coerce")
c24_mileage = pd.to_numeric(c24["mileage_km"], errors="coerce")

neutral = ["white", "silver", "grey", "gray", "black"]
c24_color_lower = c24["color"].str.lower().str.strip()

cars24_mapped = pd.DataFrame({
    "source":             "cars24",
    "make":               c24["make"].str.lower().str.strip(),
    "model":              c24["model"].str.lower().str.strip(),
    "year":               pd.to_numeric(c24["year"], errors="coerce"),
    "price":              c24["price_aed"],
    "mileage":            c24_mileage,
    "body_type":          c24["body_type"].str.lower().str.strip(),
    "cylinders":          pd.to_numeric(c24["cylinders"], errors="coerce"),
    "transmission":       c24["transmission"].str.lower().str.strip(),
    "fuel_type":          c24["fuel_type"].str.lower().str.strip(),
    "color":              c24_color_lower,
    "location":           c24["city"].str.lower().str.strip(),
    # Engineer the same numerical features
    "age":                c24_age,
    "log_mileage":        np.log1p(c24_mileage),
    "mileage_per_year":   c24_mileage / c24_age.clip(lower=1),
    # Engineer same binary features
    "is_neutral_color":   c24_color_lower.isin(neutral).astype(int),
    "is_high_performance":(pd.to_numeric(c24["cylinders"], errors="coerce") >= 8).astype(int),
    "is_premium_body":    c24["body_type"].str.lower().isin(["suv","coupe","convertible"]).astype(int),
    # Description features unknown for Cars24 — leave as NaN
    "has_sunroof":        np.nan,
    "has_leather":        np.nan,
    "has_camera":         np.nan,
    "has_cruise":         np.nan,
    "has_navigation":     np.nan,
    "has_bluetooth":      np.nan,
    # Condition derived from structured accident_free field
    "is_clean_condition": c24_is_clean,
    "has_accident_history":c24_has_accident,
    "needs_repair":       np.nan,
    "has_repainted_bumper":np.nan,
    "has_damage":         c24_has_accident,  # accident implies damage
    # Cars24-only columns — real values
    "trim":               c24["trim"].str.lower().str.strip(),
    "engine_size_l":      pd.to_numeric(c24["engine_size_l"], errors="coerce"),
    "warranty_months":    pd.to_numeric(c24["warranty_months"], errors="coerce"),
    "is_gcc_specs":       (c24["specs_region"].str.upper() == "GCC").astype(int),
})

# ── Merge ──────────────────────────────────────────────────────────────────
merged = pd.concat([kaggle_mapped, cars24_mapped], ignore_index=True)

print(f"Kaggle rows:  {len(kaggle_mapped):,}")
print(f"Cars24 rows:  {len(cars24_mapped):,}")
print(f"Total merged: {len(merged):,}")
print(f"Price range:  AED {merged.price.min():,.0f} – {merged.price.max():,.0f}")
print(f"\nMissing values per column:")
missing = merged.isnull().sum()
print(missing[missing > 0])

merged.to_csv(BASE_DIR / "uae_cars_merged.csv", index=False)
print("\nSaved: uae_cars_merged.csv")
