import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "mr1_baseline_ret3d_sweep_hold5.csv")

# -------- MR-1 parameters --------
RET3D_THRESHOLDS = [-0.03, -0.04, -0.05, -0.06, -0.07]
HOLD_DAYS = 5
ATR_MIN = 0.02

# Slippage (same model you’ve used)
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0

def fwd_roll_min(s: pd.Series, window: int) -> pd.Series:
    a = s.shift(-1)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s: pd.Series, window: int) -> pd.Series:
    a = s.shift(-1)
    return a.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]

def add_forward(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["entry_open"] = g["open"].shift(-1)
    g["exit_close"] = g["close"].shift(-HOLD_DAYS)

    low_min = fwd_roll_min(g["low"], HOLD_DAYS)
    high_max = fwd_roll_max(g["high"], HOLD_DAYS)

    g["mae"] = (low_min / g["entry_open"]) - 1.0
    g["mfe"] = (high_max / g["entry_open"]) - 1.0
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

def ensure_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure MR-1 required features exist:
      - ret3d (close/close.shift(3)-1)
      - sma10 (rolling mean of close)
    """
    g = g.copy()

    if "ret3d" not in g.columns:
        g["ret3d"] = g["close"] / g["close"].shift(3) - 1.0

    if "sma10" not in g.columns:
        g["sma10"] = g["close"].rolling(10, min_periods=10).mean()

    return g

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    # Build per-ticker so rolling features are correct
    frames = []
    for tkr, g in df.groupby("ticker", sort=False):
        g = ensure_features(g)
        g = add_forward(g)
        frames.append(g)
    full = pd.concat(frames, ignore_index=True)

    # sanity: must have atr14_pct
    if "atr14_pct" not in full.columns:
        raise RuntimeError("Missing atr14_pct in your parquet. (It should exist from your TP work.)")

    rows = []
    for thr in RET3D_THRESHOLDS:
        cond = pd.Series(True, index=full.index)

        # Oversold
        cond &= full["ret3d"].notna()
        cond &= (full["ret3d"] <= thr)

        # Below short MA
        cond &= full["sma10"].notna()
        cond &= (full["close"] < full["sma10"])

        # Volatility floor
        cond &= full["atr14_pct"].notna()
        cond &= (full["atr14_pct"] >= ATR_MIN)

        # Execution availability
        cond &= full["entry_open"].notna()
        cond &= full["exit_close"].notna()
        cond &= full["mae"].notna()
        cond &= full["mfe"].notna()

        sig = full[cond].copy()
        sig["net_ret"] = net_ret(sig["entry_open"], sig["exit_close"])

        stats = summarize(sig)
        rows.append({"ret3d_max": thr, **stats})

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}\n")
    print("MR-1 baseline sweep (hold=5D):")
    print(res.sort_values("mean_ret", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
