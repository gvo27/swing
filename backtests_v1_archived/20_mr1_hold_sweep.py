import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "mr1_hold_sweep.csv")

RET3D_THRESHOLDS = [-0.05, -0.06, -0.07]
HOLDS = [2, 3, 5, 7, 10]
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

    # Build base per ticker once
    frames = []
    for tkr, g in df.groupby("ticker", sort=False):
        g = ensure_features(g)
        g["entry_open"] = g["open"].shift(-1)
        g["year"] = g["date"].dt.year
        frames.append(g)
    full = pd.concat(frames, ignore_index=True)

    rows = []

    for hold in HOLDS:
        # forward metrics depend on hold
        tmp = full.copy()
        tmp["exit_close"] = tmp["close"].shift(-hold)

        low_min = fwd_roll_min(tmp["low"], hold)
        high_max = fwd_roll_max(tmp["high"], hold)
        tmp["mae"] = (low_min / tmp["entry_open"]) - 1.0
        tmp["mfe"] = (high_max / tmp["entry_open"]) - 1.0

        for thr in RET3D_THRESHOLDS:
            cond = pd.Series(True, index=tmp.index)
            cond &= tmp["ret3d"].notna() & (tmp["ret3d"] <= thr)
            cond &= tmp["sma10"].notna() & (tmp["close"] < tmp["sma10"])
            cond &= tmp["atr14_pct"].notna() & (tmp["atr14_pct"] >= ATR_MIN)

            cond &= tmp["entry_open"].notna() & tmp["exit_close"].notna()
            cond &= tmp["mae"].notna() & tmp["mfe"].notna()

            sig = tmp[cond].copy()
            sig["net_ret"] = net_ret(sig["entry_open"], sig["exit_close"])

            stats = summarize(sig)
            rows.append({"hold_days": hold, "ret3d_max": thr, **stats})

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}\n")
    print("Top MR-1 configs by mean_ret (n>=2000):")
    best = res[res["n"] >= 2000].sort_values("mean_ret", ascending=False).head(15)
    print(best.to_string(index=False))

if __name__ == "__main__":
    main()
