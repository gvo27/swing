import os
import numpy as np
import pandas as pd

# -----------------------------
# CONFIG (baseline Trend Pullback)
# -----------------------------
PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"

# Signal definition (RSI-free)
PULLBACK_RET5D_MAX = -0.03   # e.g., price down at least 3% over last 5 trading days
PULLBACK_RET10D_MAX = None   # set like -0.05 to also require 10D pullback, else None
REQUIRE_DOWN_DAY = True      # ret1d <= 0 (a "red day" on the signal day)
REQUIRE_TREND = True         # above SMA200 AND SMA200 slope positive

# Backtest exits: hold N trading days after signal day (enter next open)
HOLDS = [5, 10, 20]          # exit at close of t+H (entry at open t+1)

# -----------------------------
# Helpers: forward rolling min/max windows (no lookahead in signal; fwd windows only for evaluation)
# -----------------------------
def fwd_roll_min(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    """For each t, min of s[t+start_offset : t+start_offset+window-1]."""
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    """For each t, max of s[t+start_offset : t+start_offset+window-1]."""
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]


def build_signals(g: pd.DataFrame) -> pd.Series:
    # Base: require enough warmup for SMA200 slope, ret5d
    cond = pd.Series(True, index=g.index)

    if REQUIRE_TREND:
        cond &= (g["above_sma200"] == 1)
        cond &= (g["sma200_slope20"].notna())
        cond &= (g["sma200_slope20"] > 0)

    cond &= g["ret5d"].notna()
    cond &= (g["ret5d"] <= PULLBACK_RET5D_MAX)

    if PULLBACK_RET10D_MAX is not None:
        cond &= g["ret10d"].notna()
        cond &= (g["ret10d"] <= PULLBACK_RET10D_MAX)

    if REQUIRE_DOWN_DAY:
        cond &= g["ret1d"].notna()
        cond &= (g["ret1d"] <= 0)

    cond &= g["atr14_pct"].notna()
    cond &= (g["atr14_pct"] >= 0.02)

    # Also require entry day exists (next open)
    cond &= g["open"].shift(-1).notna()

    return cond


def add_forward_metrics(g: pd.DataFrame, holds: list[int]) -> pd.DataFrame:
    """
    Adds per-hold forward return, MAE, MFE based on:
      - entry at next day open (t+1 open)
      - exit at close of t+H
      - MAE/MFE over lows/highs from t+1 .. t+H (inclusive)
    """
    g = g.copy()
    g["entry_open"] = g["open"].shift(-1)

    for H in holds:
        # Exit at close of t+H (signal day = t)
        g[f"exit_close_{H}d"] = g["close"].shift(-H)

        # Window extremes from t+1 .. t+H (length H)
        low_min = fwd_roll_min(g["low"], window=H, start_offset=1)
        high_max = fwd_roll_max(g["high"], window=H, start_offset=1)

        g[f"mae_{H}d"] = (low_min / g["entry_open"]) - 1.0
        g[f"mfe_{H}d"] = (high_max / g["entry_open"]) - 1.0

        g[f"ret_{H}d"] = (g[f"exit_close_{H}d"] / g["entry_open"]) - 1.0

    return g


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Build per-ticker signals + forward metrics
    out_frames = []
    for tkr, g in df.groupby("ticker", sort=False):
        g = g.copy()

        # Ensure required columns exist
        needed = ["open","high","low","close","ret1d","ret5d","ret10d","above_sma200","sma200_slope20"]
        missing = [c for c in needed if c not in g.columns]
        if missing:
            raise RuntimeError(f"Missing columns in data for {tkr}: {missing}")

        g["signal"] = build_signals(g).astype(int)
        g = add_forward_metrics(g, HOLDS)

        out_frames.append(g)

    full = pd.concat(out_frames, ignore_index=True)

    # Build event table (one row per signal * hold)
    signals = full[full["signal"] == 1].copy()
    if signals.empty:
        raise RuntimeError("No signals found. Loosen thresholds (e.g., PULLBACK_RET5D_MAX).")

    base_cols = [
        "date","ticker","close","entry_open",
        "ret1d","ret5d","ret10d","ret20d","ret60d",
        "sma50","sma200","above_sma200","sma200_slope20",
        "atr14_pct","dd_from_52w_high"
    ]
    base_cols = [c for c in base_cols if c in signals.columns]

    event_rows = []
    for H in HOLDS:
        cols = base_cols + [f"ret_{H}d", f"mae_{H}d", f"mfe_{H}d", f"exit_close_{H}d"]
        tmp = signals[cols].copy()
        tmp["hold_days"] = H
        tmp = tmp.rename(columns={
            f"ret_{H}d": "ret",
            f"mae_{H}d": "mae",
            f"mfe_{H}d": "mfe",
            f"exit_close_{H}d": "exit_close",
        })
        event_rows.append(tmp)

    events = pd.concat(event_rows, ignore_index=True)

    # Drop incomplete forward windows
    events = events.dropna(subset=["ret","mae","mfe","exit_close","entry_open"])

    # Summaries
    def summarize(d: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": len(d),
            "win_rate": (d["ret"] > 0).mean(),
            "mean_ret": d["ret"].mean(),
            "median_ret": d["ret"].median(),
            "p25_ret": d["ret"].quantile(0.25),
            "p75_ret": d["ret"].quantile(0.75),
            "mean_mae": d["mae"].mean(),
            "mean_mfe": d["mfe"].mean(),
        })

    summary_by_hold = events.groupby("hold_days", as_index=False).apply(summarize).reset_index(drop=True)

    events["year"] = events["date"].dt.year
    summary_by_year = events.groupby(["hold_days","year"], as_index=False).apply(summarize).reset_index(drop=True)

    # Regime buckets (ATR% quartiles) for context
    events["atr_bucket"] = pd.qcut(events["atr14_pct"], 4, labels=["Q1","Q2","Q3","Q4"])
    summary_by_regime = events.groupby(["hold_days","atr_bucket"], as_index=False).apply(summarize).reset_index(drop=True)

    # Save
    events_path = os.path.join(OUT_DIR, "tp_events.csv")
    s1_path = os.path.join(OUT_DIR, "tp_summary_by_hold.csv")
    s2_path = os.path.join(OUT_DIR, "tp_summary_by_year.csv")
    s3_path = os.path.join(OUT_DIR, "tp_summary_by_regime.csv")

    events.to_csv(events_path, index=False)
    summary_by_hold.to_csv(s1_path, index=False)
    summary_by_year.to_csv(s2_path, index=False)
    summary_by_regime.to_csv(s3_path, index=False)

    print(f"Saved events:  {events_path}  rows={len(events):,}")
    print(f"Saved summary: {s1_path}")
    print(summary_by_hold.to_string(index=False))


if __name__ == "__main__":
    main()
