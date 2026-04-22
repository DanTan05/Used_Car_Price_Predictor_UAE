import pandas as pd

df = pd.read_csv("uae_used_cars_10k.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nSample:\n", df.head(3))