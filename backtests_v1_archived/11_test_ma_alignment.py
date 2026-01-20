import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "tp_ma_alignment_years_hold20.csv")

# TP_v2 core
RET5D_MAX = -0.04
ATR_MIN = 0.02
REQUIRE_DOWN_DAY = True
H = 20

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

def summarize_year(d: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(d),
        "win_net": (d["net_ret"] > 0).mean(),
        "mean_net": d["net_ret"].mean(),
        "median_net": d["net_ret"].median(),
        "mean_mae": d["mae"].mean(),
    })

def run_variant(df: pd.DataFrame, require_alignment: bool) -> pd.DataFrame:
    cond = pd.Series(True, index=df.index)

    # Trend (same as before)
    cond &= df["sma200"].notna()
    cond &= df["sma200_slope20"].notna()
    cond &= (df["sma200_slope20"] > 0)
    cond &= (df["close"] > df["sma200"])

    # NEW: MA alignment
    if require_alignment:
        cond &= df["sma50"].notna()
        cond &= (df["sma50"] > df["sma200"])

    # Pullback + down day
    cond &= df["ret5d"].notna()
    cond &= (df["ret5d"] <= RET5D_MAX)
    if REQUIRE_DOWN_DAY:
        cond &= df["ret1d"].notna()
        cond &= (df["ret1d"] <= 0)

    # ATR floor
    cond &= df["atr14_pct"].notna()
    cond &= (df["atr14_pct"] >= ATR_MIN)

    # Execution availability
    cond &= df["entry_open"].notna()
    cond &= df["exit_close"].notna()
    cond &= df["mae"].notna()

    sig = df[cond].copy()
    sig["net_ret"] = net_ret(sig["entry_open"], sig["exit_close"])
    sig["year"] = sig["date"].dt.year

    by_year = sig.groupby("year", as_index=False).apply(summarize_year).reset_index(drop=True)
    by_year["ma_alignment"] = require_alignment
    return by_year

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    # forward metrics per ticker
    frames = []
    for tkr, g in df.groupby("ticker", sort=False):
        frames.append(add_forward(g))
    full = pd.concat(frames, ignore_index=True)

    a = run_variant(full, require_alignment=False)
    b = run_variant(full, require_alignment=True)

    res = pd.concat([a, b], ignore_index=True)
    res.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}\n")

    # Print 2022 comparison + overall mean across years (n>=100)
    for flag in [False, True]:
        sub = res[res["ma_alignment"] == flag].copy()
        label = "ALIGN_ON (sma50>sma200)" if flag else "BASELINE"
        y2022 = sub[sub["year"] == 2022]
        print(f"\n{label}")
        if not y2022.empty:
            print("2022:", y2022[["n","win_net","mean_net","median_net","mean_mae"]].to_string(index=False))
        else:
            print("2022: no trades")
        sub2 = sub[sub["n"] >= 100]
        print("Overall (years with n>=100):",
              f"mean_net={sub2['mean_net'].mean():.4f}, win_net={sub2['win_net'].mean():.3f}, mean_n={sub2['n'].mean():.1f}")

if __name__ == "__main__":
    main()
