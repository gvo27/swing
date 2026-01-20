import pandas as pd

PATH = "data/qqq_dd52w.parquet"

df = pd.read_parquet(PATH)

print("=== HEAD ===")
print(df.head())

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== INFO ===")
print(df.info())

print("\n=== DATE RANGE ===")
print(df["date"].min(), "→", df["date"].max())
