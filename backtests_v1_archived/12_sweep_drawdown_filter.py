import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "tp_sweep_dd52w_hold20.csv")

# Core TP_v2
RET5D_MAX = -0.04
ATR_MIN = 0.02
REQUIRE_DOWN_DAY = True
H = 20

# Sweep: require dd_from_52w_high >= threshold (i.e., not too far below highs)
DD_THRESHOLDS = [-0.05, -0.10, -0.15, -0.20, -0.30]

# Costs/slippage
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0

def fwd_roll_min(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def add_forward(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["entry_open"] = g["open"].shift(-1)
    g["exit_close"] = g["close"].shift(-H)
    low_min = fwd_roll_min(g["low"], window=H, start_offset=1)
    g["mae"] = (low_min / g["entry_open"]) - 1.0
    return g

def net_ret(entry_open: pd.Series, exit_close: pd.Series) -> pd.Series:
    entry = entry_open.astype(float)
    exit_ = exit_close.astype(float)
    slip = SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0
    entry_fill = entry + slip
    exit_fill = exit_ - slip
    return (exit_fill / entry_fill) - 1.0

def summarize(d: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(d),
        "win_net": (d["net_ret"] > 0).mean(),
        "mean_net": d["net_ret"].mean(),
        "median_net": d["net_ret"].median(),
        "mean_mae": d["mae"].mean(),
    })

def run_threshold(full: pd.DataFrame, dd_min: float) -> pd.DataFrame:
    cond = pd.Series(True, index=full.index)

    # Trend
    cond &= full["sma200"].notna()
    cond &= full["sma200_slope20"].notna()
    cond &= (full["sma200_slope20"] > 0)
    cond &= (full["close"] > full["sma200"])

    # Pullback + down day
    cond &= full["ret5d"].notna()
    cond &= (full["ret5d"] <= RET5D_MAX)
    if REQUIRE_DOWN_DAY:
        cond &= full["ret1d"].notna()
        cond &= (full["ret1d"] <= 0)

    # ATR
    cond &= full["atr14_pct"].notna()
    cond &= (full["atr14_pct"] >= ATR_MIN)

    # NEW: drawdown filter
    cond &= full["dd_from_52w_high"].notna()
    cond &= (full["dd_from_52w_high"] >= dd_min)

    # execution
    cond &= full["entry_open"].notna()
    cond &= full["exit_close"].notna()
    cond &= full["mae"].notna()

    sig = full[cond].copy()
    sig["net_ret"] = net_ret(sig["entry_open"], sig["exit_close"])
    sig["year"] = sig["date"].dt.year

    by_year = sig.groupby("year", as_index=False).apply(summarize).reset_index(drop=True)
    by_year["dd_min"] = dd_min
    return by_year

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    frames = []
    for tkr, g in df.groupby("ticker", sort=False):
        frames.append(add_forward(g))
    full = pd.concat(frames, ignore_index=True)

    all_rows = []
    for dd_min in DD_THRESHOLDS:
        by_year = run_threshold(full, dd_min)
        all_rows.append(by_year)

    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}\n")

    for dd_min in DD_THRESHOLDS:
        sub = res[res["dd_min"] == dd_min].copy()
        y2022 = sub[sub["year"] == 2022]
        print(f"\ndd_from_52w_high >= {dd_min:.2f}")
        if not y2022.empty:
            print("2022:", y2022[["n","win_net","mean_net","median_net","mean_mae"]].to_string(index=False))
        else:
            print("2022: no trades")

        sub2 = sub[sub["n"] >= 100]
        print("Overall (years with n>=100):",
              f"mean_net={sub2['mean_net'].mean():.4f}, win_net={sub2['win_net'].mean():.3f}, mean_n={sub2['n'].mean():.1f}")

if __name__ == "__main__":
    main()
