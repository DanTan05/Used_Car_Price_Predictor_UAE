import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "data/uae_used_cars_10k.csv")

# Standardize text columns to lowercase
df["Make"]  = df["Make"].str.lower().str.strip()
df["Model"] = df["Model"].str.lower().str.strip()
df["Color"] = df["Color"].str.lower().str.strip()

# Cylinders column comes in as string — coerce to number first
df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors="coerce")
df["Cylinders"] = df["Cylinders"].fillna(df["Cylinders"].median())

# Cap at AED 300k: error analysis showed cars above AED 350k drive 70%+ of
# total MAE despite being only ~15% of rows. The model had too few examples
# of them to learn well. This focuses the model on the realistic buyer range.
df = df[(df["Price"] >= 5_000) & (df["Price"] <= 300_000)]

# Remove mileage outliers
df = df[(df["Mileage"] >= 0) & (df["Mileage"] <= 600_000)]

df.to_csv(BASE_DIR / "uae_used_cars_clean.csv", index=False)
print(f"Clean dataset: {len(df)} rows")
print(f"Price range: AED {df['Price'].min():,.0f} – {df['Price'].max():,.0f}")
print(f"Price median: AED {df['Price'].median():,.0f}")
