import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"

# -----------------------------
# Slippage (Robinhood style)
# -----------------------------
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0
COMMISSION_PER_TRADE_USD = 0.0

def net_ret_from_prices(entry_open: pd.Series, exit_close: pd.Series) -> pd.Series:
    slip = SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0
    entry_fill = entry_open.astype(float) + slip
    exit_fill = exit_close.astype(float) - slip
    net = (exit_fill / entry_fill) - 1.0
    if COMMISSION_PER_TRADE_USD != 0.0:
        net = net - (2.0 * COMMISSION_PER_TRADE_USD) / entry_fill
    return net

# -----------------------------
# Strategy definitions
# -----------------------------
# TP_v2 (locked)
TP_HOLD = 20
TP_RET5D_MAX = -0.04
TP_ATR_MIN = 0.02

# MR-1A (locked candidate)
MR_HOLD = 10
MR_RET3D_MAX = -0.07
MR_ATR_MIN = 0.02

def ensure_mr_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    if "ret3d" not in g.columns:
        g["ret3d"] = g["close"] / g["close"].shift(3) - 1.0
    if "sma10" not in g.columns:
        g["sma10"] = g["close"].rolling(10, min_periods=10).mean()
    return g

def tp_v2_signal(g: pd.DataFrame) -> pd.Series:
    cond = pd.Series(True, index=g.index)
    cond &= g["sma200"].notna()
    cond &= g["sma200_slope20"].notna()
    cond &= (g["sma200_slope20"] > 0)
    cond &= (g["close"] > g["sma200"])
    cond &= g["ret5d"].notna() & (g["ret5d"] <= TP_RET5D_MAX)
    cond &= g["ret1d"].notna() & (g["ret1d"] <= 0)
    cond &= g["atr14_pct"].notna() & (g["atr14_pct"] >= TP_ATR_MIN)

    # Need entry at t+1 open and exit at t+TP_HOLD close
    cond &= g["open"].shift(-1).notna()
    cond &= g["close"].shift(-TP_HOLD).notna()
    return cond

def mr_1a_signal(g: pd.DataFrame) -> pd.Series:
    cond = pd.Series(True, index=g.index)
    cond &= g["ret3d"].notna() & (g["ret3d"] <= MR_RET3D_MAX)
    cond &= g["sma10"].notna() & (g["close"] < g["sma10"])
    cond &= g["atr14_pct"].notna() & (g["atr14_pct"] >= MR_ATR_MIN)

    # Need entry at t+1 open and exit at t+MR_HOLD close
    cond &= g["open"].shift(-1).notna()
    cond &= g["close"].shift(-MR_HOLD).notna()
    return cond

# -----------------------------
# Streams helpers
# -----------------------------
def build_events(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two event tables: tp_events, mr_events
    Columns:
      strategy, ticker, signal_date, entry_date, exit_date, entry_open, exit_close, net_ret
    """
    tp_rows = []
    mr_rows = []

    for tkr, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        g = ensure_mr_features(g)

        # TP events
        tp_sig = tp_v2_signal(g)
        if tp_sig.any():
            sig_i = np.where(tp_sig.to_numpy(bool))[0]
            ev = pd.DataFrame({
                "strategy": "TP_v2",
                "ticker": tkr,
                "signal_date": g.loc[sig_i, "date"].to_numpy(),
                "entry_date": g.loc[sig_i + 1, "date"].to_numpy(),
                "exit_date": g.loc[sig_i + TP_HOLD, "date"].to_numpy(),
                "entry_open": g.loc[sig_i + 1, "open"].to_numpy(float),
                "exit_close": g.loc[sig_i + TP_HOLD, "close"].to_numpy(float),
            })
            ev["net_ret"] = net_ret_from_prices(ev["entry_open"], ev["exit_close"])
            tp_rows.append(ev)

        # MR events
        mr_sig = mr_1a_signal(g)
        if mr_sig.any():
            sig_i = np.where(mr_sig.to_numpy(bool))[0]
            ev = pd.DataFrame({
                "strategy": "MR_1A",
                "ticker": tkr,
                "signal_date": g.loc[sig_i, "date"].to_numpy(),
                "entry_date": g.loc[sig_i + 1, "date"].to_numpy(),
                "exit_date": g.loc[sig_i + MR_HOLD, "date"].to_numpy(),
                "entry_open": g.loc[sig_i + 1, "open"].to_numpy(float),
                "exit_close": g.loc[sig_i + MR_HOLD, "close"].to_numpy(float),
            })
            ev["net_ret"] = net_ret_from_prices(ev["entry_open"], ev["exit_close"])
            mr_rows.append(ev)

    tp_events = pd.concat(tp_rows, ignore_index=True) if tp_rows else pd.DataFrame()
    mr_events = pd.concat(mr_rows, ignore_index=True) if mr_rows else pd.DataFrame()
    return tp_events, mr_events

def diff_count_series(dates_index: pd.DatetimeIndex, starts: pd.Series, ends: pd.Series) -> pd.Series:
    """
    Active-trade count series using difference array:
    +1 at start date index
    -1 at (end date + 1 trading day index in our index) -> implemented as end_idx+1
    Assumes starts/ends are in dates_index.
    Active includes both start and end dates.
    """
    n = len(dates_index)
    diff = np.zeros(n + 1, dtype=int)

    start_idx = dates_index.get_indexer(starts.to_numpy())
    end_idx = dates_index.get_indexer(ends.to_numpy())

    # drop any -1 (missing dates) just in case
    ok = (start_idx >= 0) & (end_idx >= 0)
    start_idx = start_idx[ok]
    end_idx = end_idx[ok]

    np.add.at(diff, start_idx, 1)
    np.add.at(diff, end_idx + 1, -1)  # safe because diff has n+1
    active = np.cumsum(diff[:-1])
    return pd.Series(active, index=dates_index)

def daily_realized_series(dates_index: pd.DatetimeIndex, exit_dates: pd.Series, net_rets: pd.Series) -> pd.Series:
    """
    Realized P&L stream: sum of net_ret on the exit day (unit per trade).
    """
    s = pd.Series(net_rets.to_numpy(float), index=pd.to_datetime(exit_dates.to_numpy()))
    # group by date and reindex to full calendar
    realized = s.groupby(level=0).sum()
    return realized.reindex(dates_index, fill_value=0.0)

def daily_count_series(dates_index: pd.DatetimeIndex, dts: pd.Series) -> pd.Series:
    c = pd.Series(1, index=pd.to_datetime(dts.to_numpy()))
    out = c.groupby(level=0).sum()
    return out.reindex(dates_index, fill_value=0).astype(int)

def summarize_concurrency(active: pd.Series) -> dict:
    arr = active.to_numpy()
    return {
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }

# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    tp_events, mr_events = build_events(df)

    if tp_events.empty:
        raise RuntimeError("No TP_v2 events found. Check your dataset/features.")
    if mr_events.empty:
        raise RuntimeError("No MR_1A events found. Check your dataset/features.")

    # Save events
    tp_path = os.path.join(OUT_DIR, "tp_v2_events_hold20.csv")
    mr_path = os.path.join(OUT_DIR, "mr_1a_events_hold10.csv")
    tp_events.to_csv(tp_path, index=False)
    mr_events.to_csv(mr_path, index=False)

    # Unified calendar
    all_dates = pd.to_datetime(
        pd.concat([df["date"], tp_events["entry_date"], tp_events["exit_date"], mr_events["entry_date"], mr_events["exit_date"]])
        .dropna().unique()
    )
    dates_index = pd.DatetimeIndex(sorted(all_dates))

    # TP daily streams
    tp_active = diff_count_series(dates_index, tp_events["entry_date"], tp_events["exit_date"])
    tp_entries = daily_count_series(dates_index, tp_events["entry_date"])
    tp_exits = daily_count_series(dates_index, tp_events["exit_date"])
    tp_realized = daily_realized_series(dates_index, tp_events["exit_date"], tp_events["net_ret"])

    # MR daily streams
    mr_active = diff_count_series(dates_index, mr_events["entry_date"], mr_events["exit_date"])
    mr_entries = daily_count_series(dates_index, mr_events["entry_date"])
    mr_exits = daily_count_series(dates_index, mr_events["exit_date"])
    mr_realized = daily_realized_series(dates_index, mr_events["exit_date"], mr_events["net_ret"])

    # Combined
    both_active = (tp_active > 0) & (mr_active > 0)
    any_active = (tp_active > 0) | (mr_active > 0)

    # Correlation of realized streams (daily)
    corr = float(pd.Series(tp_realized).corr(pd.Series(mr_realized)))

    # Build daily dataframe and save
    daily = pd.DataFrame({
        "tp_active": tp_active,
        "tp_entries": tp_entries,
        "tp_exits": tp_exits,
        "tp_realized_unit": tp_realized,

        "mr_active": mr_active,
        "mr_entries": mr_entries,
        "mr_exits": mr_exits,
        "mr_realized_unit": mr_realized,

        "both_active": both_active.astype(int),
        "any_active": any_active.astype(int),
        "combined_active": (tp_active + mr_active),
        "combined_realized_unit": (tp_realized + mr_realized),
    }, index=dates_index)

    daily_path = os.path.join(OUT_DIR, "daily_streams_tp_v2_and_mr_1a.csv")
    daily.to_csv(daily_path, index_label="date")

    # Print report
    print(f"Saved events:\n  {tp_path}\n  {mr_path}")
    print(f"Saved daily streams:\n  {daily_path}\n")
    print(f"Event counts: TP_v2={len(tp_events):,}  MR_1A={len(mr_events):,}")
    print(f"Realized daily stream correlation (TP vs MR): {corr:.4f}\n")

    tp_conc = summarize_concurrency(tp_active)
    mr_conc = summarize_concurrency(mr_active)
    comb_conc = summarize_concurrency(daily["combined_active"])

    print("Concurrency (active trades) percentiles:")
    print(f"  TP_v2:  p50={tp_conc['p50']:.0f}  p75={tp_conc['p75']:.0f}  p90={tp_conc['p90']:.0f}  p95={tp_conc['p95']:.0f}  max={tp_conc['max']:.0f}  mean={tp_conc['mean']:.2f}")
    print(f"  MR_1A:  p50={mr_conc['p50']:.0f}  p75={mr_conc['p75']:.0f}  p90={mr_conc['p90']:.0f}  p95={mr_conc['p95']:.0f}  max={mr_conc['max']:.0f}  mean={mr_conc['mean']:.2f}")
    print(f"  COMB:   p50={comb_conc['p50']:.0f}  p75={comb_conc['p75']:.0f}  p90={comb_conc['p90']:.0f}  p95={comb_conc['p95']:.0f}  max={comb_conc['max']:.0f}  mean={comb_conc['mean']:.2f}\n")

    # Overlap stats
    pct_both_when_any = 100.0 * daily["both_active"].sum() / max(1, daily["any_active"].sum())
    pct_days_both = 100.0 * daily["both_active"].mean()
    print("Overlap:")
    print(f"  % of days BOTH active (of all days): {pct_days_both:.2f}%")
    print(f"  % of active days where BOTH active: {pct_both_when_any:.2f}%")

if __name__ == "__main__":
    main()
