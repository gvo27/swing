import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "mr1_year_focus_hold10.csv")

RET3D_THRESHOLDS = [-0.05, -0.06, -0.07]
HOLD_DAYS = 10
ATR_MIN = 0.02
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0

def fwd_roll_min(s: pd.Series, window: int) -> pd.Series:
    a = s.shift(-1)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s: pd.Series, window: int) -> pd.Series:
    a = s.shift(-1)
    return a.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]

def ensure_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    if "ret3d" not in g.columns:
        g["ret3d"] = g["close"] / g["close"].shift(3) - 1.0
    if "sma10" not in g.columns:
        g["sma10"] = g["close"].rolling(10, min_periods=10).mean()
    return g

def net_ret(entry_open: pd.Series, exit_close: pd.Series) -> pd.Series:
    slip = SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0
    entry_fill = entry_open + slip
    exit_fill = exit_close - slip
    return (exit_fill / entry_fill) - 1.0

def summarize(d: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(d),
        "win_rate": (d["net_ret"] > 0).mean() if len(d) else np.nan,
        "mean_ret": d["net_ret"].mean() if len(d) else np.nan,
        "median_ret": d["net_ret"].median() if len(d) else np.nan,
        "p25": d["net_ret"].quantile(0.25) if len(d) else np.nan,
        "p75": d["net_ret"].quantile(0.75) if len(d) else np.nan,
        "mean_mae": d["mae"].mean() if len(d) else np.nan,
        "mean_mfe": d["mfe"].mean() if len(d) else np.nan,
    })

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    frames = []
    for tkr, g in df.groupby("ticker", sort=False):
        g = ensure_features(g)
        g["entry_open"] = g["open"].shift(-1)
        g["exit_close"] = g["close"].shift(-HOLD_DAYS)
        g["year"] = g["date"].dt.year

        low_min = fwd_roll_min(g["low"], HOLD_DAYS)
        high_max = fwd_roll_max(g["high"], HOLD_DAYS)
        g["mae"] = (low_min / g["entry_open"]) - 1.0
        g["mfe"] = (high_max / g["entry_open"]) - 1.0
        frames.append(g)

    full = pd.concat(frames, ignore_index=True)

    rows = []
    for thr in RET3D_THRESHOLDS:
        cond = pd.Series(True, index=full.index)
        cond &= full["ret3d"].notna() & (full["ret3d"] <= thr)
        cond &= full["sma10"].notna() & (full["close"] < full["sma10"])
        cond &= full["atr14_pct"].notna() & (full["atr14_pct"] >= ATR_MIN)
        cond &= full["entry_open"].notna() & full["exit_close"].notna()
        cond &= full["mae"].notna() & full["mfe"].notna()

        sig = full[cond].copy()
        sig["net_ret"] = net_ret(sig["entry_open"], sig["exit_close"])

        by_year = sig.groupby("year", as_index=False).apply(summarize).reset_index(drop=True)
        by_year["ret3d_max"] = thr
        rows.append(by_year)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}\n")

    focus_years = [2018, 2020, 2022, 2025]
    view = out[out["year"].isin(focus_years)].sort_values(["ret3d_max","year"])
    print("MR-1 year focus (hold=10D, net):")
    print(view[["ret3d_max","year","n","win_rate","mean_ret","median_ret","mean_mae","mean_mfe"]].to_string(index=False))

if __name__ == "__main__":
    main()
