import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"

# -----------------------------
# Slippage assumptions
# -----------------------------
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0
COMMISSION_PER_TRADE_USD = 0.0

def net_ret_from_prices(entry_open, exit_close):
    slip = SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0
    entry_fill = entry_open.astype(float) + slip
    exit_fill = exit_close.astype(float) - slip
    r = (exit_fill / entry_fill) - 1.0
    if COMMISSION_PER_TRADE_USD != 0.0:
        r -= (2.0 * COMMISSION_PER_TRADE_USD) / entry_fill
    return r

# -----------------------------
# Locked strategy definitions
# -----------------------------
# TP_v2
TP_HOLD = 20
TP_RET5D_MAX = -0.04
TP_ATR_MIN = 0.02

# MR-1A
MR_HOLD = 10
MR_RET3D_MAX = -0.07
MR_ATR_MIN = 0.02

# EB-1A
EB_HOLD = 10
EB_GAP_MIN = 0.04
EB_ATR_MIN = 0.02

# -----------------------------
# Utilities
# -----------------------------
def fwd_roll_min(s, window):
    a = s.shift(-1)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s, window):
    a = s.shift(-1)
    return a.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]

def ensure_mr_features(g):
    g = g.copy()
    if "ret3d" not in g.columns:
        g["ret3d"] = g["close"] / g["close"].shift(3) - 1.0
    if "sma10" not in g.columns:
        g["sma10"] = g["close"].rolling(10, min_periods=10).mean()
    return g

# -----------------------------
# Signal builders
# -----------------------------
def build_tp_events(g):
    cond = (
        g["sma200"].notna()
        & g["sma200_slope20"].notna()
        & (g["sma200_slope20"] > 0)
        & (g["close"] > g["sma200"])
        & g["ret5d"].notna()
        & (g["ret5d"] <= TP_RET5D_MAX)
        & g["ret1d"].notna()
        & (g["ret1d"] <= 0)
        & g["atr14_pct"].notna()
        & (g["atr14_pct"] >= TP_ATR_MIN)
        & g["open"].shift(-1).notna()
        & g["close"].shift(-TP_HOLD).notna()
    )

    idx = np.where(cond)[0]
    if len(idx) == 0:
        return pd.DataFrame()

    ev = pd.DataFrame({
        "strategy": "TP_v2",
        "entry_date": g.loc[idx + 1, "date"].to_numpy(),
        "exit_date": g.loc[idx + TP_HOLD, "date"].to_numpy(),
        "entry_open": g.loc[idx + 1, "open"].to_numpy(),
        "exit_close": g.loc[idx + TP_HOLD, "close"].to_numpy(),
    })
    ev["net_ret"] = net_ret_from_prices(ev["entry_open"], ev["exit_close"])
    return ev

def build_mr_events(g):
    g = ensure_mr_features(g)

    cond = (
        g["ret3d"].notna()
        & (g["ret3d"] <= MR_RET3D_MAX)
        & g["sma10"].notna()
        & (g["close"] < g["sma10"])
        & g["atr14_pct"].notna()
        & (g["atr14_pct"] >= MR_ATR_MIN)
        & g["open"].shift(-1).notna()
        & g["close"].shift(-MR_HOLD).notna()
    )

    idx = np.where(cond)[0]
    if len(idx) == 0:
        return pd.DataFrame()

    ev = pd.DataFrame({
        "strategy": "MR_1A",
        "entry_date": g.loc[idx + 1, "date"].to_numpy(),
        "exit_date": g.loc[idx + MR_HOLD, "date"].to_numpy(),
        "entry_open": g.loc[idx + 1, "open"].to_numpy(),
        "exit_close": g.loc[idx + MR_HOLD, "close"].to_numpy(),
    })
    ev["net_ret"] = net_ret_from_prices(ev["entry_open"], ev["exit_close"])
    return ev

def build_eb_events(g):
    g = g.copy()
    g["prev_close"] = g["close"].shift(1)
    g["gap"] = (g["open"] / g["prev_close"]) - 1.0
    g["day_ret"] = (g["close"] / g["open"]) - 1.0

    cond = (
        g["gap"].notna()
        & (g["gap"] >= EB_GAP_MIN)
        & (g["day_ret"] >= 0)
        & g["atr14_pct"].notna()
        & (g["atr14_pct"] >= EB_ATR_MIN)
        & g["sma200"].notna()
        & (g["close"] > g["sma200"])
        & g["open"].shift(-1).notna()
        & g["close"].shift(-EB_HOLD).notna()
    )

    idx = np.where(cond)[0]
    if len(idx) == 0:
        return pd.DataFrame()

    ev = pd.DataFrame({
        "strategy": "EB_1A",
        "entry_date": g.loc[idx + 1, "date"].to_numpy(),
        "exit_date": g.loc[idx + EB_HOLD, "date"].to_numpy(),
        "entry_open": g.loc[idx + 1, "open"].to_numpy(),
        "exit_close": g.loc[idx + EB_HOLD, "close"].to_numpy(),
    })
    ev["net_ret"] = net_ret_from_prices(ev["entry_open"], ev["exit_close"])
    return ev

# -----------------------------
# Stream helpers
# -----------------------------
def diff_active_series(dates, starts, ends):
    diff = np.zeros(len(dates) + 1)
    s_idx = dates.get_indexer(starts)
    e_idx = dates.get_indexer(ends)
    ok = (s_idx >= 0) & (e_idx >= 0)
    np.add.at(diff, s_idx[ok], 1)
    np.add.at(diff, e_idx[ok] + 1, -1)
    return pd.Series(np.cumsum(diff[:-1]), index=dates)

def realized_series(dates, exit_dates, rets):
    s = pd.Series(rets.values, index=pd.to_datetime(exit_dates))
    return s.groupby(level=0).sum().reindex(dates, fill_value=0.0)

def summarize_active(s):
    a = s.values
    return {
        "p50": np.quantile(a, 0.50),
        "p75": np.quantile(a, 0.75),
        "p90": np.quantile(a, 0.90),
        "p95": np.quantile(a, 0.95),
        "max": np.max(a),
        "mean": np.mean(a),
    }

# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    tp, mr, eb = [], [], []

    for _, g in df.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        tp.append(build_tp_events(g))
        mr.append(build_mr_events(g))
        eb.append(build_eb_events(g))

    tp = pd.concat(tp, ignore_index=True)
    mr = pd.concat(mr, ignore_index=True)
    eb = pd.concat(eb, ignore_index=True)

    all_dates = pd.to_datetime(
        pd.concat([df["date"], tp["entry_date"], mr["entry_date"], eb["entry_date"]])
        .dropna().unique()
    )
    dates = pd.DatetimeIndex(sorted(all_dates))

    tp_act = diff_active_series(dates, tp["entry_date"], tp["exit_date"])
    mr_act = diff_active_series(dates, mr["entry_date"], mr["exit_date"])
    eb_act = diff_active_series(dates, eb["entry_date"], eb["exit_date"])

    tp_pnl = realized_series(dates, tp["exit_date"], tp["net_ret"])
    mr_pnl = realized_series(dates, mr["exit_date"], mr["net_ret"])
    eb_pnl = realized_series(dates, eb["exit_date"], eb["net_ret"])

    print("\n=== Realized daily return correlations ===")
    print(f"TP vs MR: {tp_pnl.corr(mr_pnl):.4f}")
    print(f"TP vs EB: {tp_pnl.corr(eb_pnl):.4f}")
    print(f"MR vs EB: {mr_pnl.corr(eb_pnl):.4f}")

    print("\n=== Concurrency percentiles (active trades) ===")
    print("TP_v2:", summarize_active(tp_act))
    print("MR_1A:", summarize_active(mr_act))
    print("EB_1A:", summarize_active(eb_act))
    print("COMBINED:", summarize_active(tp_act + mr_act + eb_act))

if __name__ == "__main__":
    main()
