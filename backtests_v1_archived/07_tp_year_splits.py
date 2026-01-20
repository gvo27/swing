import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_BY_YEAR = os.path.join(OUT_DIR, "tp_year_summary.csv")
OUT_BY_YEAR_HOLD = os.path.join(OUT_DIR, "tp_year_summary_by_hold.csv")

# --- TP_v2 signal (locked from your work) ---
RET5D_MAX = -0.04
ATR_MIN = 0.02
REQUIRE_TREND = True
REQUIRE_DOWN_DAY = True

HOLDS = [5, 10, 20]

# --- Costs / slippage (Robinhood-style) ---
COMMISSION_PER_TRADE_USD = 0.0

USE_CENTS_SLIPPAGE = True
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0  # $0.02/share each side

# Alternative bps model (set USE_CENTS_SLIPPAGE=False to use this)
SLIPPAGE_BPS_PER_SIDE = 0.0


def fwd_roll_min(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]


def build_signal(g: pd.DataFrame) -> pd.Series:
    cond = pd.Series(True, index=g.index)

    if REQUIRE_TREND:
        cond &= (g["above_sma200"] == 1)
        cond &= g["sma200_slope20"].notna()
        cond &= (g["sma200_slope20"] > 0)

    cond &= g["ret5d"].notna()
    cond &= (g["ret5d"] <= RET5D_MAX)

    if REQUIRE_DOWN_DAY:
        cond &= g["ret1d"].notna()
        cond &= (g["ret1d"] <= 0)

    cond &= g["atr14_pct"].notna()
    cond &= (g["atr14_pct"] >= ATR_MIN)

    cond &= g["open"].shift(-1).notna()
    return cond


def add_forward_metrics(g: pd.DataFrame, holds: list[int]) -> pd.DataFrame:
    g = g.copy()
    g["entry_open"] = g["open"].shift(-1)

    for H in holds:
        g[f"exit_close_{H}d"] = g["close"].shift(-H)

        low_min = fwd_roll_min(g["low"], window=H, start_offset=1)
        high_max = fwd_roll_max(g["high"], window=H, start_offset=1)

        g[f"ret_{H}d"] = (g[f"exit_close_{H}d"] / g["entry_open"]) - 1.0
        g[f"mae_{H}d"] = (low_min / g["entry_open"]) - 1.0
        g[f"mfe_{H}d"] = (high_max / g["entry_open"]) - 1.0

    return g


def apply_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: entry_open, exit_close
    Adds: ret (gross), net_ret
    """
    df = df.copy()
    entry = df["entry_open"].astype(float)
    exit_ = df["exit_close"].astype(float)

    gross = (exit_ / entry) - 1.0

    if USE_CENTS_SLIPPAGE:
        slip = SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0
        entry_fill = entry + slip
        exit_fill = exit_ - slip
        net = (exit_fill / entry_fill) - 1.0
    else:
        bps = SLIPPAGE_BPS_PER_SIDE / 10000.0
        net = gross - 2.0 * bps

    if COMMISSION_PER_TRADE_USD != 0.0:
        net = net - (2.0 * COMMISSION_PER_TRADE_USD) / entry

    df["ret"] = gross
    df["net_ret"] = net
    return df


def summarize(d: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(d),
        "win_gross": (d["ret"] > 0).mean(),
        "mean_gross": d["ret"].mean(),
        "median_gross": d["ret"].median(),
        "win_net": (d["net_ret"] > 0).mean(),
        "mean_net": d["net_ret"].mean(),
        "median_net": d["net_ret"].median(),
        "mean_mae": d["mae"].mean(),
        "mean_mfe": d["mfe"].mean(),
    })


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Precompute forward metrics once per ticker
    frames = []
    for tkr, g in df.groupby("ticker", sort=False):
        g = add_forward_metrics(g, HOLDS)
        g["signal"] = build_signal(g).astype(int)
        frames.append(g)
    full = pd.concat(frames, ignore_index=True)

    sig = full[full["signal"] == 1].copy()
    if sig.empty:
        raise RuntimeError("No signals found with current parameters.")

    sig["year"] = sig["date"].dt.year

    event_rows = []
    for H in HOLDS:
        ev = sig[["date","year","ticker","entry_open", f"exit_close_{H}d", f"mae_{H}d", f"mfe_{H}d"]].copy()
        ev = ev.rename(columns={f"exit_close_{H}d": "exit_close", f"mae_{H}d": "mae", f"mfe_{H}d": "mfe"})
        ev["hold_days"] = H
        ev = ev.dropna(subset=["entry_open","exit_close","mae","mfe"])
        ev = apply_costs(ev)
        event_rows.append(ev)

    events = pd.concat(event_rows, ignore_index=True)

    # Summary by year+hold
    by_year_hold = events.groupby(["hold_days","year"], as_index=False).apply(summarize).reset_index(drop=True)

    # Overall by year (pool holds if you want; here we keep hold-specific file + also an overall hold=20 focus)
    by_year = by_year_hold.copy()

    by_year.to_csv(OUT_BY_YEAR_HOLD, index=False)

    # Also create a convenience file/table for hold=20 only (often your main swing horizon)
    by_year_20 = by_year_hold[by_year_hold["hold_days"] == 20].copy()
    by_year_20.to_csv(OUT_BY_YEAR, index=False)

    print(f"Saved: {OUT_BY_YEAR_HOLD}")
    print(f"Saved: {OUT_BY_YEAR} (hold=20 only)\n")

    # Print: Hold=20 year table (net)
    show = by_year_20.sort_values("year")
    print("Hold=20D by year (net):")
    print(show[["year","n","win_net","mean_net","median_net","mean_mae","mean_mfe"]].to_string(index=False))

    # Quick worst/best years by mean_net (hold=20)
    best = show.sort_values("mean_net", ascending=False).head(3)
    worst = show.sort_values("mean_net", ascending=True).head(3)
    print("\nBest 3 years (hold=20, mean_net):")
    print(best[["year","n","win_net","mean_net","median_net","mean_mae"]].to_string(index=False))
    print("\nWorst 3 years (hold=20, mean_net):")
    print(worst[["year","n","win_net","mean_net","median_net","mean_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
