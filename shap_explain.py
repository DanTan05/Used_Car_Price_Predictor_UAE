import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from pathlib import Path

BASE_DIR = Path(__file__).parent

FEATURE_NAMES = [
    "make", "model", "age", "log_mileage", "mileage_per_year", "cylinders",
    "transmission", "fuel_type", "body_type", "location",
    "trim", "engine_size_l", "warranty_months", "is_gcc_specs",
    "has_sunroof", "has_leather", "has_camera", "has_cruise",
    "has_navigation", "has_bluetooth",
    "is_clean_condition", "has_accident_history", "needs_repair",
    "has_repainted_bumper", "has_damage",
    "is_neutral_color", "is_high_performance", "is_premium_body",
]

model = xgb.XGBRegressor()
model.load_model(BASE_DIR / "models/xgboost_model.json")

X_test = np.load(BASE_DIR / "X_test.npy")
y_test = np.load(BASE_DIR / "y_test.npy")

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)  # shape: (n_samples, n_features)
shap_values.feature_names = FEATURE_NAMES

print(f"SHAP values computed for {len(X_test):,} test samples, {len(FEATURE_NAMES)} features")

# --- Global: mean |SHAP| bar chart ---
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
order = np.argsort(mean_abs_shap)[::-1]

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(
    [FEATURE_NAMES[i] for i in order],
    mean_abs_shap[order],
    color="steelblue",
)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP value| (log-price scale)")
ax.set_title("Global Feature Importance — XGBoost (SHAP)")
plt.tight_layout()
plt.savefig(BASE_DIR / "outputs/shap_bar.png", dpi=150)
plt.close()
print("Saved shap_bar.png")

# Print top-10 to console
print("\nTop-10 features by mean |SHAP|:")
for i in order[:10]:
    print(f"  {FEATURE_NAMES[i]:25s}  {mean_abs_shap[i]:.4f}")

# --- Global: beeswarm (shows direction + spread, not just magnitude) ---
fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.savefig(BASE_DIR / "outputs/shap_beeswarm.png", dpi=150)
plt.close()
print("Saved shap_beeswarm.png")

# --- Local: waterfall for 3 individual predictions ---
actual_prices = np.expm1(y_test)
sample_indices = [
    np.argmin(actual_prices),            # cheapest car in test set
    np.argmax(actual_prices),            # most expensive
    len(X_test) // 2,                    # middle sample
]
labels = ["cheapest", "most_expensive", "middle"]

for idx, label in zip(sample_indices, labels):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
    predicted_price = np.expm1(model.predict(X_test[[idx]])[0])
    plt.title(f"SHAP Waterfall — {label} car  (predicted AED {predicted_price:,.0f}, actual AED {actual_prices[idx]:,.0f})")
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"outputs/shap_waterfall_{label}.png", dpi=150)
    plt.close()
    print(f"Saved shap_waterfall_{label}.png  (predicted AED {predicted_price:,.0f} | actual AED {actual_prices[idx]:,.0f})")
