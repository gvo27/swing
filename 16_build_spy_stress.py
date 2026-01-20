import pandas as pd
import numpy as np

QQQ_DD52W_PATH = "data/qqq_dd52w.parquet"   # you already have this
OUT_PATH = "data/market_stress.parquet"

def main():
    q = pd.read_parquet(QQQ_DD52W_PATH).copy()
    q["date"] = pd.to_datetime(q["date"]).dt.normalize()

    # If you already have SPY somewhere else, swap input file + column name here.
    # For now we use QQQ as a proxy only if you truly don't have SPY.
    # BEST is SPY; if you do have SPY prices, point this script at that file.
    if "qqq_close" in q.columns:
        close = q["qqq_close"].astype(float)
        px_name = "qqq"
    elif "close" in q.columns:
        close = q["close"].astype(float)
        px_name = "close"
    else:
        raise RuntimeError(f"Can't find a close column. Columns: {q.columns.tolist()}")

    q[f"{px_name}_ret20d"] = close.pct_change(20)
    q["market_stress"] = q[f"{px_name}_ret20d"] <= -0.10

    out = q[["date", f"{px_name}_ret20d", "market_stress"]].dropna().sort_values("date")
    out.to_parquet(OUT_PATH, index=False)

    print(f"Saved {OUT_PATH} with last date {out['date'].max().date()}")

if __name__ == "__main__":
    main()
