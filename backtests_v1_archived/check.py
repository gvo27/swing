import pandas as pd

df = pd.read_parquet("data/sp100_daily_features.parquet")

print(df.head())
print(df.tail())
print("Tickers:", df["ticker"].nunique())
print("Date range:", df["date"].min(), "→", df["date"].max())
