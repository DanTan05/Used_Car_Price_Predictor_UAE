"""
Edit the CAR dict below, then run:
    .venv/Scripts/python predict.py
"""
import json
import numpy as np
import xgboost as xgb
from pathlib import Path

# ── Edit this ──────────────────────────────────────────────────────────────────
CAR = {
    # Core
    "make":         "toyota",       # lowercase, e.g. "toyota", "mercedes-benz", "nissan"
    "model":        "camry",        # lowercase model name
    "trim":         "se",           # lowercase trim, or "" if unknown
    "year":         2020,
    "mileage":      60000,          # km
    "cylinders":    4,
    "engine_size_l": 2.5,           # litres, or None if unknown
    "transmission": "automatic",    # "automatic" or "manual"
    "fuel_type":    "gasoline",     # "gasoline", "diesel", "hybrid", "electric"
    "body_type":    "sedan",        # "sedan", "suv", "hatchback", "coupe", "convertible", etc.
    "location":     "dubai",        # city, lowercase
    "color":        "white",        # for is_neutral_color flag
    "warranty_months": 0,
    "is_gcc_specs": 1,              # 1 = yes, 0 = no

    # Features (1 = yes, 0 = no)
    "has_sunroof":   0,
    "has_leather":   1,
    "has_camera":    1,
    "has_cruise":    0,
    "has_navigation": 0,
    "has_bluetooth": 1,

    # Condition — pick exactly one that applies
    "is_clean_condition":   1,
    "has_accident_history": 0,
    "needs_repair":         0,
    "has_repainted_bumper": 0,
    "has_damage":           0,
}
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "models/encodings.json") as f:
    enc = json.load(f)

label_encoders = enc["label_encoders"]
target_maps    = enc["target_maps"]
global_mean    = enc["global_mean"]

model = xgb.XGBRegressor()
model.load_model(BASE_DIR / "models/xgboost_model.json")

NEUTRAL_COLORS   = {"white", "silver", "grey", "gray", "black"}
PREMIUM_BODIES   = {"suv", "coupe", "convertible"}

age             = 2025 - CAR["year"]
log_mileage     = np.log1p(CAR["mileage"])
mileage_per_year = CAR["mileage"] / max(age, 1)
is_neutral_color  = int(CAR["color"].lower() in NEUTRAL_COLORS)
is_high_performance = int(CAR["cylinders"] >= 8)
is_premium_body   = int(CAR["body_type"].lower() in PREMIUM_BODIES)

def encode_label(col, value):
    mapping = label_encoders.get(col, {})
    val = str(value).lower()
    if val in mapping:
        return mapping[val]
    # Fallback: midpoint of known range
    print(f"  Warning: '{value}' not seen for '{col}', using fallback encoding 0")
    return 0

def encode_target(col, value):
    mapping = target_maps.get(col, {})
    val = str(value).lower()
    if val in mapping:
        return mapping[val]
    print(f"  Warning: '{value}' not seen for '{col}', using global mean")
    return global_mean

features = [
    encode_target("make",  CAR["make"]),
    encode_target("model", CAR["model"]),
    age,
    log_mileage,
    mileage_per_year,
    CAR["cylinders"],
    encode_label("transmission", CAR["transmission"]),
    encode_label("fuel_type",    CAR["fuel_type"]),
    encode_label("body_type",    CAR["body_type"]),
    encode_label("location",     CAR["location"]),
    encode_target("trim",        CAR["trim"] if CAR["trim"] else ""),
    CAR["engine_size_l"] if CAR["engine_size_l"] is not None else 0,
    CAR["warranty_months"],
    CAR["is_gcc_specs"],
    CAR["has_sunroof"],
    CAR["has_leather"],
    CAR["has_camera"],
    CAR["has_cruise"],
    CAR["has_navigation"],
    CAR["has_bluetooth"],
    CAR["is_clean_condition"],
    CAR["has_accident_history"],
    CAR["needs_repair"],
    CAR["has_repainted_bumper"],
    CAR["has_damage"],
    is_neutral_color,
    is_high_performance,
    is_premium_body,
]

X = np.array(features, dtype=float).reshape(1, -1)
log_pred = model.predict(X)[0]
predicted_price = np.expm1(log_pred)

print(f"\n{'-' * 40}")
print(f"  {CAR['year']} {CAR['make'].title()} {CAR['model'].title()}")
print(f"  {CAR['mileage']:,} km | {CAR['cylinders']} cyl | {CAR['transmission']}")
print(f"{'-' * 40}")
print(f"  Predicted price:  AED {predicted_price:,.0f}")
print(f"{'-' * 40}\n")
