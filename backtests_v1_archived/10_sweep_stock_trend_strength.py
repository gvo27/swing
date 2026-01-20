import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "tp_sweep_strength_years_hold20.csv")

# --- Core TP_v2 signal (no market filter) ---
RET5D_MAX = -0.04
ATR_MIN = 0.02
REQUIRE_DOWN_DAY = True

HOLDS = [20]

# Sweep: how far above SMA200 must price be?
MIN_ABOVE_SMA200_LIST = [0.00, 0.02, 0.03, 0.05]

# Costs/slippage (same as yours)
USE_CENTS_SLIPPAGE = True
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0


def fwd_roll_min(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]


def add_forward_metrics(g: pd.DataFrame, holds: list[int]) -> pd.DataFrame:
    g = g.copy()
    g["entry_open"] = g["open"].shift(-1)
    for H in holds:
        g[f"exit_close_{H}d"] = g["close"].shift(-H)
        low_min = fwd_roll_min(g["low"], window=H, start_offset=1)
        g[f"mae_{H}d"] = (low_min / g["entry_open"]) - 1.0
    return g


def apply_costs(entry_open: pd.Series, exit_close: pd.Series) -> pd.Series:
    entry = entry_open.astype(float)
    exit_ = exit_close.astype(float)
    gross = (exit_ / entry) - 1.0
    if USE_CENTS_SLIPPAGE:
        slip = SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0
        entry_fill = entry + slip
        exit_fill = exit_ - slip
        net = (exit_fill / entry_fill) - 1.0
        return net
    return gross


def summarize_year(d: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(d),
        "win_net": (d["net_ret"] > 0).mean(),
        "mean_net": d["net_ret"].mean(),
        "median_net": d["net_ret"].median(),
        "mean_mae": d["mae"].mean(),
    })


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    # Precompute forward metrics once per ticker
    frames = []
    for tkr, g in df.groupby("ticker", sort=False):
        frames.append(add_forward_metrics(g, HOLDS))
    full = pd.concat(frames, ignore_index=True)

    rows = []
    H = 20

    for min_above in MIN_ABOVE_SMA200_LIST:
        g = full.copy()

        # Strength measure: distance above SMA200
        g["dist_sma200"] = (g["close"] / g["sma200"]) - 1.0

        # Signal
        cond = pd.Series(True, index=g.index)
        cond &= g["sma200"].notna()
        cond &= (g["dist_sma200"].notna())
        cond &= (g["dist_sma200"] >= min_above)

        cond &= g["sma200_slope20"].notna()
        cond &= (g["sma200_slope20"] > 0)

        cond &= g["ret5d"].notna()
        cond &= (g["ret5d"] <= RET5D_MAX)

        if REQUIRE_DOWN_DAY:
            cond &= g["ret1d"].notna()
            cond &= (g["ret1d"] <= 0)

        cond &= g["atr14_pct"].notna()
        cond &= (g["atr14_pct"] >= ATR_MIN)

        cond &= g["entry_open"].notna()
        cond &= g[f"exit_close_{H}d"].notna()
        cond &= g[f"mae_{H}d"].notna()

        sig = g[cond].copy()
        sig["year"] = sig["date"].dt.year
        sig["mae"] = sig[f"mae_{H}d"]
        sig["net_ret"] = apply_costs(sig["entry_open"], sig[f"exit_close_{H}d"])

        by_year = sig.groupby("year", as_index=False).apply(summarize_year).reset_index(drop=True)
        by_year["min_above_sma200"] = min_above
        rows.append(by_year)

    res = pd.concat(rows, ignore_index=True)
    res.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}\n")

    # Print a focused view: 2022 + overall mean across years (excluding tiny-n years)
    for min_above in MIN_ABOVE_SMA200_LIST:
        sub = res[res["min_above_sma200"] == min_above].copy()
        sub = sub.sort_values("year")
        y2022 = sub[sub["year"] == 2022]
        print(f"\nmin_above_sma200 = {min_above:.2f}")
        if not y2022.empty:
            print("2022:", y2022[["n","win_net","mean_net","median_net","mean_mae"]].to_string(index=False))
        else:
            print("2022: no trades")

        # overall across years with n>=100
        sub2 = sub[sub["n"] >= 100]
        print("Overall (years with n>=100):",
              f"mean_net={sub2['mean_net'].mean():.4f}, win_net={sub2['win_net'].mean():.3f}, mean_n={sub2['n'].mean():.1f}")

if __name__ == "__main__":
    main()
