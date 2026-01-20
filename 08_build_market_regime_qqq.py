import os
import pandas as pd
import yfinance as yf

OUT_DIR = "data"
OUT_PATH = os.path.join(OUT_DIR, "qqq_regime.parquet")

START = "2014-01-01"
TICKER = "QQQ"

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df has MultiIndex columns, flatten to single level.
    Example: ('Close','QQQ') -> 'close_qqq' OR ('QQQ','Close')-> 'qqq_close'
    We'll just join non-empty parts with underscore.
    """
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            parts = [str(x).strip() for x in col if x not in (None, "", " ")]
            new_cols.append("_".join(parts))
        df = df.copy()
        df.columns = new_cols
    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    q = yf.download(TICKER, start=START, auto_adjust=False, progress=False)

    if q is None or q.empty:
        raise RuntimeError("Failed to download QQQ data.")

    # Flatten MultiIndex columns if needed
    q = flatten_columns(q)

    # Move index to a column
    q = q.reset_index()

    # Normalize column names
    q.columns = [str(c).strip().lower().replace(" ", "_") for c in q.columns]

    # Find the date column (it might be 'date', 'datetime', etc.)
    date_col = None
    for cand in ["date", "datetime", "index"]:
        if cand in q.columns:
            date_col = cand
            break

    if date_col is None:
        # last-resort: find any column that contains 'date'
        for c in q.columns:
            if "date" in c:
                date_col = c
                break

    if date_col is None:
        raise RuntimeError(f"Could not find date column. Columns: {q.columns.tolist()}")

    if date_col != "date":
        q = q.rename(columns={date_col: "date"})

    q["date"] = pd.to_datetime(q["date"])

    # Find the close column robustly
    # After flattening, you might see 'close', or 'close_qqq', or 'qqq_close'
    close_col = None
    for cand in ["close", "close_qqq", "qqq_close"]:
        if cand in q.columns:
            close_col = cand
            break
    if close_col is None:
        # fallback: any column that starts with 'close'
        for c in q.columns:
            if c.startswith("close"):
                close_col = c
                break
    if close_col is None:
        raise RuntimeError(f"Could not find close column. Columns: {q.columns.tolist()}")

    # Build regime features
    q["qqq_close"] = q[close_col].astype(float)
    q["qqq_sma200"] = sma(q["qqq_close"], 200)
    q["qqq_above_sma200"] = (q["qqq_close"] > q["qqq_sma200"]).astype("int")
    q["qqq_sma200_slope20"] = q["qqq_sma200"].pct_change(20)

    out = q[["date", "qqq_close", "qqq_sma200", "qqq_above_sma200", "qqq_sma200_slope20"]].copy()
    out.to_parquet(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}  rows={len(out):,}  date range={out['date'].min().date()}→{out['date'].max().date()}")

if __name__ == "__main__":
    main()
