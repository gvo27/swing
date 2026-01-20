import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"

# -----------------------------
# TP_v3 (LOCKED)
# -----------------------------
RET5D_MAX = -0.04
ATR_MIN = 0.02
DD52W_MIN = -0.10
REQUIRE_DOWN_DAY = True
REQUIRE_TREND = True

HOLDS = [5, 10, 20]

# -----------------------------
# Costs / slippage (Robinhood-style)
# -----------------------------
COMMISSION_PER_TRADE_USD = 0.0
USE_CENTS_SLIPPAGE = True
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0
SLIPPAGE_BPS_PER_SIDE = 0.0


def fwd_roll_min(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]


def add_forward_metrics(g: pd.DataFrame, holds: list[int]) -> pd.DataFrame:
    g = g.copy()
    g["entry_open"] = g["open"].shift(-1)

    for H in holds:
        g[f"exit_close_{H}d"] = g["close"].shift(-H)

        low_min = fwd_roll_min(g["low"], window=H, start_offset=1)
        high_max = fwd_roll_max(g["high"], window=H, start_offset=1)

        g[f"mae_{H}d"] = (low_min / g["entry_open"]) - 1.0
        g[f"mfe_{H}d"] = (high_max / g["entry_open"]) - 1.0
        g[f"ret_{H}d"] = (g[f"exit_close_{H}d"] / g["entry_open"]) - 1.0

    return g


def build_signal(g: pd.DataFrame) -> pd.Series:
    cond = pd.Series(True, index=g.index)

    if REQUIRE_TREND:
        cond &= g["sma200"].notna()
        cond &= g["sma200_slope20"].notna()
        cond &= (g["sma200_slope20"] > 0)
        cond &= (g["close"] > g["sma200"])

    cond &= g["ret5d"].notna()
    cond &= (g["ret5d"] <= RET5D_MAX)

    if REQUIRE_DOWN_DAY:
        cond &= g["ret1d"].notna()
        cond &= (g["ret1d"] <= 0)

    cond &= g["atr14_pct"].notna()
    cond &= (g["atr14_pct"] >= ATR_MIN)

    cond &= g["dd_from_52w_high"].notna()
    cond &= (g["dd_from_52w_high"] >= DD52W_MIN)

    cond &= g["open"].shift(-1).notna()
    return cond


def apply_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects: entry_open, exit_close
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
        "p25_net": d["net_ret"].quantile(0.25),
        "p75_net": d["net_ret"].quantile(0.75),
        "mean_mae": d["mae"].mean(),
        "mean_mfe": d["mfe"].mean(),
    })


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    frames = []
    for tkr, g in df.groupby("ticker", sort=False):
        g = add_forward_metrics(g, HOLDS)
        g["signal"] = build_signal(g).astype(int)
        frames.append(g)

    full = pd.concat(frames, ignore_index=True)
    sig = full[full["signal"] == 1].copy()

    if sig.empty:
        raise RuntimeError("No signals for TP_v3. Check filter thresholds.")

    sig["year"] = sig["date"].dt.year

    # Build events table (one row per signal per hold)
    event_rows = []
    for H in HOLDS:
        ev = sig[[
            "date","year","ticker","entry_open",
            f"exit_close_{H}d", f"ret_{H}d", f"mae_{H}d", f"mfe_{H}d",
            "ret1d","ret5d","atr14_pct","dd_from_52w_high",
            "sma200_slope20"
        ]].copy()

        ev = ev.rename(columns={
            f"exit_close_{H}d": "exit_close",
            f"ret_{H}d": "ret_gross_pre",
            f"mae_{H}d": "mae",
            f"mfe_{H}d": "mfe",
        })
        ev["hold_days"] = H

        ev = ev.dropna(subset=["entry_open","exit_close","mae","mfe"])
        ev = apply_costs(ev)
        event_rows.append(ev)

    events = pd.concat(event_rows, ignore_index=True)

    # Save events
    events_path = os.path.join(OUT_DIR, "tp_v3_events_hold5_10_20.csv")
    events.to_csv(events_path, index=False)

    # Summary by hold
    summary_by_hold = events.groupby("hold_days", as_index=False).apply(summarize).reset_index(drop=True)
    summary_path = os.path.join(OUT_DIR, "tp_v3_summary_by_hold.csv")
    summary_by_hold.to_csv(summary_path, index=False)

    # Year-by-year for hold=20 (net)
    ev20 = events[events["hold_days"] == 20].copy()
    year20 = ev20.groupby("year", as_index=False).apply(summarize).reset_index(drop=True)
    year20_path = os.path.join(OUT_DIR, "tp_v3_year_hold20.csv")
    year20.to_csv(year20_path, index=False)

    print(f"Saved events:  {events_path}  rows={len(events):,}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved year20:  {year20_path}\n")

    print("TP_v3 Summary by hold (NET + GROSS):")
    print(summary_by_hold[[
        "hold_days","n",
        "win_gross","mean_gross","median_gross",
        "win_net","mean_net","median_net",
        "p25_net","p75_net",
        "mean_mae","mean_mfe"
    ]].to_string(index=False))

    print("\nTP_v3 Hold=20 by year (NET):")
    print(year20.sort_values("year")[["year","n","win_net","mean_net","median_net","mean_mae","mean_mfe"]].to_string(index=False))


if __name__ == "__main__":
    main()
