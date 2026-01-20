import os
import pandas as pd
import yfinance as yf

OUT_DIR = "data_live"
OUT_PATH = os.path.join(OUT_DIR, "qqq_dd52w.parquet")
START = "2014-01-01"
TICKER = "QQQ"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    q = yf.download(TICKER, start=START, auto_adjust=False, progress=False)
    if q is None or q.empty:
        raise RuntimeError("Failed to download QQQ data.")

    # Flatten potential MultiIndex columns
    if isinstance(q.columns, pd.MultiIndex):
        q.columns = ["_".join([str(x).strip() for x in col if str(x).strip() not in ("", "None")]).lower()
                     for col in q.columns]
    else:
        q.columns = [str(c).strip().lower() for c in q.columns]

    q = q.reset_index()
    q.columns = [str(c).strip().lower().replace(" ", "_") for c in q.columns]

    # robust date + close
    date_col = "date" if "date" in q.columns else ("index" if "index" in q.columns else None)
    if date_col is None:
        for c in q.columns:
            if "date" in c:
                date_col = c
                break
    if date_col is None:
        raise RuntimeError(f"Could not find date column. Columns: {q.columns.tolist()}")
    if date_col != "date":
        q = q.rename(columns={date_col: "date"})
    q["date"] = pd.to_datetime(q["date"])

    close_col = None
    for cand in ["close", "close_qqq", "qqq_close"]:
        if cand in q.columns:
            close_col = cand
            break
    if close_col is None:
        for c in q.columns:
            if c.startswith("close"):
                close_col = c
                break
    if close_col is None:
        raise RuntimeError(f"Could not find close column. Columns: {q.columns.tolist()}")

    q["qqq_close"] = q[close_col].astype(float)

    # 52-week high approx = 252 trading days
    q["qqq_52w_high"] = q["qqq_close"].rolling(252, min_periods=252).max()
    q["qqq_dd_from_52w_high"] = (q["qqq_close"] / q["qqq_52w_high"]) - 1.0

    out = q[["date", "qqq_close", "qqq_dd_from_52w_high"]].copy()
    out.to_parquet(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH} rows={len(out):,} range={out['date'].min().date()}→{out['date'].max().date()}")

if __name__ == "__main__":
    main()
