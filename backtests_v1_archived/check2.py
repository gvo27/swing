import pandas as pd
df = pd.read_parquet("data/sp100_daily_features.parquet")
print(sorted([c for c in df.columns if "ret" in c or "sma" in c])[:200])
