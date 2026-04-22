import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).parent

X_train = np.load(BASE_DIR / "X_train.npy")
X_test  = np.load(BASE_DIR / "X_test.npy")
y_train = np.load(BASE_DIR / "y_train.npy")
y_test  = np.load(BASE_DIR / "y_test.npy")

# Train Random Forest (best model from Phase 2)
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)

actual    = np.expm1(y_test)
predicted = np.expm1(preds)
errors    = np.abs(actual - predicted)

# Build an analysis dataframe using the features dataset
# We reload features to attach human-readable columns back to the test rows
features_df = pd.read_csv(BASE_DIR / "uae_used_cars_features.csv")
features_df["log_price"] = np.log1p(features_df["Price"])

from sklearn.model_selection import train_test_split
_, test_df = train_test_split(features_df, test_size=0.2, random_state=42)
test_df = test_df.copy()
test_df["actual_price"]    = actual
test_df["predicted_price"] = predicted
test_df["abs_error"]       = errors
test_df["pct_error"]       = errors / actual * 100

print("=" * 55)
print("OVERALL")
print(f"  MAE:    AED {errors.mean():,.0f}")
print(f"  Median: AED {np.median(errors):,.0f}")
print(f"  90th %: AED {np.percentile(errors, 90):,.0f}")

# --- Where are the biggest errors? ---
print("\n" + "=" * 55)
print("MAE BY PRICE BUCKET")
test_df["price_bucket"] = pd.cut(
    test_df["actual_price"],
    bins=[0, 50_000, 100_000, 200_000, 350_000, 600_000],
    labels=["<50k", "50k-100k", "100k-200k", "200k-350k", "350k-600k"],
)
bucket_mae = test_df.groupby("price_bucket", observed=True)["abs_error"].agg(["mean", "count"])
bucket_mae.columns = ["MAE (AED)", "Count"]
bucket_mae["MAE (AED)"] = bucket_mae["MAE (AED)"].map("{:,.0f}".format)
print(bucket_mae)

print("\n" + "=" * 55)
print("TOP 10 MAKES BY MAE")
make_mae = (
    test_df.groupby("Make")["abs_error"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "MAE", "count": "Count"})
    .query("Count >= 10")
    .sort_values("MAE", ascending=False)
    .head(10)
)
make_mae["MAE"] = make_mae["MAE"].map("AED {:,.0f}".format)
print(make_mae)

print("\n" + "=" * 55)
print("MAE BY AGE BUCKET")
test_df["age_bucket"] = pd.cut(
    test_df["age"],
    bins=[0, 3, 6, 10, 15, 30],
    labels=["0-3 yrs", "3-6 yrs", "6-10 yrs", "10-15 yrs", "15+ yrs"],
)
age_mae = test_df.groupby("age_bucket", observed=True)["abs_error"].agg(["mean", "count"])
age_mae.columns = ["MAE (AED)", "Count"]
age_mae["MAE (AED)"] = age_mae["MAE (AED)"].map("{:,.0f}".format)
print(age_mae)

print("\n" + "=" * 55)
print("WORST 10 INDIVIDUAL PREDICTIONS")
worst = (
    test_df[["Make", "Model", "actual_price", "predicted_price", "abs_error", "age", "Mileage"]]
    .sort_values("abs_error", ascending=False)
    .head(10)
)
worst["actual_price"]    = worst["actual_price"].map("AED {:,.0f}".format)
worst["predicted_price"] = worst["predicted_price"].map("AED {:,.0f}".format)
worst["abs_error"]       = worst["abs_error"].map("AED {:,.0f}".format)
print(worst.to_string(index=False))
