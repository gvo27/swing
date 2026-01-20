import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
QQQ_PATH = "data/qqq_dd52w.parquet"   # columns: date, qqq_close

# --- v1.1 params (yours) ---
TP_HOLD = 20
TP_RET5D_MAX = -0.04
TP_ATR_MIN = 0.02

MR_HOLD = 10
MR_RET3D_MAX = -0.07
MR_ATR_MIN = 0.02

# --- v2 BO candidate (locked from your strong-gate test) ---
BO_LOOKBACK = 100
BO_RETEST_K = 3
BO_BAND = 0.005
BO_HOLD = 20

# Execution assumptions (match your sim)
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0

def entry_fill(open_px: float) -> float:
    return float(open_px) + (SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0)

def exit_fill(close_px: float) -> float:
    return float(close_px) - (SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0)

def build_qqq_regime_maps(qqq_path: str) -> tuple[dict, dict]:
    """
    Returns:
      risk_on:        QQQ close > SMA200
      risk_on_strong: (QQQ close > SMA200) AND (SMA50 > SMA200)
    """
    q = pd.read_parquet(qqq_path).copy()
    q["date"] = pd.to_datetime(q["date"])
    if "qqq_close" not in q.columns:
        raise RuntimeError(f"QQQ file missing 'qqq_close'. Columns: {q.columns.tolist()}")

    q = q.sort_values("date").reset_index(drop=True)
    q["sma50"] = q["qqq_close"].rolling(50, min_periods=50).mean()
    q["sma200"] = q["qqq_close"].rolling(200, min_periods=200).mean()

    q["risk_on"] = q["qqq_close"] > q["sma200"]
    q["risk_on_strong"] = (q["qqq_close"] > q["sma200"]) & (q["sma50"] > q["sma200"])

    risk_on = {pd.Timestamp(d).normalize(): bool(v) for d, v in zip(q["date"], q["risk_on"])}
    risk_on_strong = {pd.Timestamp(d).normalize(): bool(v) for d, v in zip(q["date"], q["risk_on_strong"])}

    return risk_on, risk_on_strong

# -----------------------------
# SIGNALS (paste your v1.1 TP_v2 exact logic here if it has more filters)
# -----------------------------

def tp_signal(row) -> bool:
    # Minimal TP_v2 (based on your params). If your v1.1 TP has extra filters, add them here.
    if not np.isfinite(row.get("ret5d", np.nan)): 
        return False
    if row["ret5d"] > TP_RET5D_MAX:
        return False
    if not np.isfinite(row.get("atr14_pct", np.nan)):
        return False
    if row["atr14_pct"] < TP_ATR_MIN:
        return False
    return True

def mr_signal(row) -> bool:
    if not np.isfinite(row.get("ret3d", np.nan)):
        return False
    if row["ret3d"] > MR_RET3D_MAX:
        return False
    if not np.isfinite(row.get("sma10", np.nan)):
        return False
    if not np.isfinite(row.get("close", np.nan)):
        return False
    if row["close"] >= row["sma10"]:
        return False
    if not np.isfinite(row.get("atr14_pct", np.nan)):
        return False
    if row["atr14_pct"] < MR_ATR_MIN:
        return False
    return True

# -----------------------------
# BO event builder (same as your sweep)
# -----------------------------

def fwd_roll_min(s: pd.Series, window: int) -> pd.Series:
    return s.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s: pd.Series, window: int) -> pd.Series:
    return s.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]

def build_bo_events_for_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True).copy()
    prior_high = g["high"].shift(1).rolling(BO_LOOKBACK, min_periods=BO_LOOKBACK).max()
    breakout = (g["close"] > prior_high * (1.0 + BO_BAND)) & prior_high.notna()
    if not breakout.any():
        return pd.DataFrame()

    rows = []
    b_idx = np.where(breakout.values)[0]

    for bi in b_idx:
        lvl = float(prior_high.iloc[bi])
        if not np.isfinite(lvl) or lvl <= 0:
            continue

        start = bi + 1
        end = min(len(g) - 1, bi + BO_RETEST_K)
        if start > end:
            continue

        window = g.iloc[start:end+1]
        cond = (window["low"] <= lvl * (1.0 + BO_BAND)) & (window["close"] >= lvl)
        if not cond.any():
            continue

        ri = int(cond.idxmax())
        entry_i = ri + 1
        exit_i = entry_i + BO_HOLD
        if exit_i >= len(g):
            continue

        rows.append({
            "ticker": g.loc[entry_i, "ticker"],
            "signal_date": g.loc[ri, "date"],
            "entry_date": g.loc[entry_i, "date"],
            "exit_date": g.loc[exit_i, "date"],
            "entry_open": float(g.loc[entry_i, "open"]),
            "exit_close": float(g.loc[exit_i, "close"]),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()

# -----------------------------
# Generic fixed-hold event builder (TP/MR)
# -----------------------------

def build_fixed_hold_events_for_ticker(g: pd.DataFrame, hold_days: int, signal_fn, gate_map=None, gate_default=True) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True).copy()
    rows = []
    for i in range(len(g) - (hold_days + 1)):
        signal_date = g.loc[i, "date"]
        if gate_map is not None:
            ok = gate_map.get(pd.Timestamp(signal_date).normalize(), gate_default)
            if not ok:
                continue

        row = g.loc[i]
        if not signal_fn(row):
            continue

        entry_i = i + 1
        exit_i = entry_i + hold_days
        if exit_i >= len(g):
            continue

        rows.append({
            "ticker": g.loc[entry_i, "ticker"],
            "signal_date": signal_date,
            "entry_date": g.loc[entry_i, "date"],
            "exit_date": g.loc[exit_i, "date"],
            "entry_open": float(g.loc[entry_i, "open"]),
            "exit_close": float(g.loc[exit_i, "close"]),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# -----------------------------
# Convert events -> daily return stream (equal-weight across active trades)
# -----------------------------

def build_daily_stream(df_all: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Returns:
      daily_ret: Series indexed by date (normalized), value = avg return across active trades (0 if none)
      conc:      Series indexed by date, value = number of active trades that day
    """
    if events.empty:
        idx = pd.Index(sorted(pd.to_datetime(df_all["date"]).dt.normalize().unique()))
        return pd.Series(0.0, index=idx), pd.Series(0.0, index=idx)

    # Prepare a fast lookup: (ticker -> its date->close series, date->open series)
    df_all = df_all.copy()
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["d"] = df_all["date"].dt.normalize()

    by_t = {}
    for tkr, g in df_all.groupby("ticker", sort=False):
        g = g.sort_values("date")
        by_t[tkr] = {
            "d": g["d"].to_numpy(),
            "close": g["close"].to_numpy(),
            "open": g["open"].to_numpy(),
        }

    # Universe calendar
    cal = pd.Index(sorted(df_all["d"].unique()))
    daily_sum = pd.Series(0.0, index=cal)
    daily_cnt = pd.Series(0.0, index=cal)

    for _, tr in events.iterrows():
        tkr = tr["ticker"]
        if tkr not in by_t:
            continue
        d_arr = by_t[tkr]["d"]
        c_arr = by_t[tkr]["close"]
        o_arr = by_t[tkr]["open"]

        entry_d = pd.Timestamp(tr["entry_date"]).normalize()
        exit_d = pd.Timestamp(tr["exit_date"]).normalize()

        # Find index positions in this ticker array
        # (assumes dates exist; if not, skip)
        try:
            entry_pos = int(np.where(d_arr == entry_d)[0][0])
            exit_pos = int(np.where(d_arr == exit_d)[0][0])
        except Exception:
            continue

        if exit_pos <= entry_pos:
            continue

        eopen = float(o_arr[entry_pos])
        efill = entry_fill(eopen)
        if not np.isfinite(efill) or efill <= 0:
            continue

        # Build "value" series for the trade across days entry..exit
        # value[t] = close[t]/efill except on exit day use exit_fill(exit_close)/efill
        dates = d_arr[entry_pos:exit_pos+1]
        closes = c_arr[entry_pos:exit_pos+1].astype(float)

        vals = closes / efill
        # apply exit slippage on last day
        xfill = exit_fill(float(tr["exit_close"]))
        vals[-1] = xfill / efill

        # daily returns from values
        rets = np.empty_like(vals)
        rets[0] = vals[0] - 1.0
        rets[1:] = (vals[1:] / vals[:-1]) - 1.0

        # accumulate into daily average stream
        for d, r in zip(dates, rets):
            if d in daily_sum.index:
                daily_sum.loc[d] += float(r)
                daily_cnt.loc[d] += 1.0

    # average returns, fill zeros when no active trades
    daily_ret = daily_sum.copy()
    mask = daily_cnt > 0
    daily_ret.loc[mask] = daily_sum.loc[mask] / daily_cnt.loc[mask]
    daily_ret.loc[~mask] = 0.0

    conc = daily_cnt.copy()
    return daily_ret, conc

def conc_stats(conc: pd.Series) -> dict:
    a = conc.to_numpy()
    return {
        "p50": float(np.quantile(a, 0.50)),
        "p75": float(np.quantile(a, 0.75)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }

def main():
    df = pd.read_parquet(PARQUET_PATH)

    

    def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        # handles MultiIndex/tuple columns like ("date","") or ("close","aapl")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(x) for x in tup if str(x) not in ("", "None")]).strip("_")
                          for tup in df.columns.to_list()]
        else:
            df.columns = [("_".join([str(x) for x in c if str(x) not in ("", "None")]).strip("_")
                           if isinstance(c, tuple) else str(c))
                          for c in df.columns]
        return df

    df = flatten_columns(df)

    # --- convert wide → long (same pattern you used before) ---
    if "ticker" not in df.columns:
        # identify feature base names
        feature_cols = [c for c in df.columns if "_" in c and c != "date"]
        base_names = sorted(set(c.rsplit("_", 1)[0] for c in feature_cols))

        records = []
        for base in base_names:
            sub = df[["date"] + [c for c in df.columns if c.startswith(base + "_")]].copy()
            sub.columns = ["date"] + [c.replace(base + "_", "") for c in sub.columns[1:]]
            sub["ticker"] = base.upper()
            records.append(sub)

        df = pd.concat(records, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    print("\n=== DEBUG: columns present ===")
    print(sorted(df.columns.tolist()))

    print("\n=== DEBUG: required column check ===")
    for c in ["ret3d", "sma10", "ret5d", "atr14_pct", "close"]:
        print(c, "->", c in df.columns)

    print("\n=== DEBUG: sample rows ===")
    print(df.head(3))
    print(df.sample(3, random_state=1))

    df["ret3d"] = df.groupby("ticker")["close"].pct_change(3)
    df["sma10"] = df.groupby("ticker")["close"].transform(lambda s: s.rolling(10, min_periods=10).mean())

    needed = {"date","ticker","open","high","low","close","atr14_pct","ret5d","ret3d","sma10"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in parquet: {missing}")

    qqq_risk_on, qqq_risk_on_strong = build_qqq_regime_maps(QQQ_PATH)

    # --- build events ---
    tp_events_list = []
    mr_events_list = []
    bo_events_list = []

    for tkr, g in df.groupby("ticker", sort=False):
        # TP gated by risk_on (QQQ > SMA200) on SIGNAL day
        tp_e = build_fixed_hold_events_for_ticker(
            g, TP_HOLD, tp_signal, gate_map=qqq_risk_on, gate_default=True
        )
        if not tp_e.empty:
            tp_events_list.append(tp_e)

        # MR always on
        mr_e = build_fixed_hold_events_for_ticker(
            g, MR_HOLD, mr_signal, gate_map=None
        )
        if not mr_e.empty:
            mr_events_list.append(mr_e)

        # BO built from BO logic, then apply strong gate on SIGNAL day
        bo_e = build_bo_events_for_ticker(g)
        if not bo_e.empty:
            bo_e["ok"] = bo_e["signal_date"].map(lambda d: qqq_risk_on_strong.get(pd.Timestamp(d).normalize(), False))
            bo_e = bo_e[bo_e["ok"] == True].drop(columns=["ok"])
            if not bo_e.empty:
                bo_events_list.append(bo_e)

    tp_events = pd.concat(tp_events_list, ignore_index=True) if tp_events_list else pd.DataFrame()
    mr_events = pd.concat(mr_events_list, ignore_index=True) if mr_events_list else pd.DataFrame()
    bo_events = pd.concat(bo_events_list, ignore_index=True) if bo_events_list else pd.DataFrame()

    print("\nEvent counts:")
    print(f"  TP_v2: {len(tp_events):,}")
    print(f"  MR_1A: {len(mr_events):,}")
    print(f"  BO_1A (strong gated): {len(bo_events):,}")

    # --- build daily streams ---
    tp_ret, tp_conc = build_daily_stream(df, tp_events)
    mr_ret, mr_conc = build_daily_stream(df, mr_events)
    bo_ret, bo_conc = build_daily_stream(df, bo_events)

    comb_ret = tp_ret + mr_ret + bo_ret
    comb_conc = tp_conc + mr_conc + bo_conc

    # --- correlations (use full calendar with zeros, as in your prior runs) ---
    corr = pd.DataFrame({
        "TP": tp_ret,
        "MR": mr_ret,
        "BO": bo_ret
    }).corr()

    print("\n=== Realized daily return correlations (equal-weight active-trade stream) ===")
    print(corr.to_string())

    # --- concurrency stats ---
    print("\n=== Concurrency percentiles (active trades) ===")
    print("TP_v2:", conc_stats(tp_conc))
    print("MR_1A:", conc_stats(mr_conc))
    print("BO_1A:", conc_stats(bo_conc))
    print("COMBINED:", conc_stats(comb_conc))

    # --- overlap diagnostics ---
    tp_active = tp_conc > 0
    mr_active = mr_conc > 0
    bo_active = bo_conc > 0

    print("\n=== Overlap (% of all days) ===")
    print("TP & MR both active:", float((tp_active & mr_active).mean()))
    print("TP & BO both active:", float((tp_active & bo_active).mean()))
    print("MR & BO both active:", float((mr_active & bo_active).mean()))
    print("All three active:", float((tp_active & mr_active & bo_active).mean()))

    print("Sanity columns:", sorted(df.columns.tolist()))
    print("Sample rows:")
    print(df.head())

if __name__ == "__main__":
    main()
