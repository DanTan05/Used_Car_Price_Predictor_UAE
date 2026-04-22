import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
import xgboost as xgb

BASE_DIR = Path(__file__).parent
X_train = np.load(BASE_DIR / "X_train.npy")
X_test  = np.load(BASE_DIR / "X_test.npy")
y_train = np.load(BASE_DIR / "y_train.npy")
y_test  = np.load(BASE_DIR / "y_test.npy")

# Combine train+test for cross-validation (CV uses all the data)
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    # Parameters from RandomizedSearchCV (50 iterations, 5-fold CV)
    "XGBoost": xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=7,
        gamma=0.3,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        verbosity=0,
    ),
}

# --- Single train/test split results (same as before) ---
print("=" * 60)
print("SINGLE SPLIT RESULTS (80/20)")
print("=" * 60)
single_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds     = model.predict(X_test)
    actual    = np.expm1(y_test)
    predicted = np.expm1(preds)
    mae = mean_absolute_error(actual, predicted)
    r2  = r2_score(y_test, preds)
    single_results[name] = {"MAE (AED)": f"{mae:,.0f}", "R²": round(r2, 3)}
    print(f"{name:25s} | MAE: AED {mae:,.0f} | R²: {r2:.3f}")

# --- 5-fold cross-validation ---
# Each fold trains on 4/5 of data and tests on the remaining 1/5.
# We repeat this 5 times with different splits and average the MAE.
# This gives a much more reliable estimate than a single split.
print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    fold_maes = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
        X_tr, X_val = X_all[train_idx], X_all[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]

        model.fit(X_tr, y_tr)
        preds     = model.predict(X_val)
        actual    = np.expm1(y_val)
        predicted = np.expm1(preds)
        fold_maes.append(mean_absolute_error(actual, predicted))

    mean_mae = np.mean(fold_maes)
    std_mae  = np.std(fold_maes)
    cv_results[name] = {"CV MAE": f"{mean_mae:,.0f}", "Std": f"±{std_mae:,.0f}"}
    print(f"{name:25s} | CV MAE: AED {mean_mae:,.0f}  (±{std_mae:,.0f})")

print("\nSingle split vs CV comparison:")
print(pd.DataFrame(single_results).T.join(pd.DataFrame(cv_results).T))

# Re-fit XGBoost on full training split and save for SHAP analysis
xgb_model = models["XGBoost"]
xgb_model.fit(X_train, y_train)
xgb_model.save_model(BASE_DIR / "models/xgboost_model.json")
print("\nSaved xgboost_model.json")

