import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
QQQ_PATH = "data/qqq_dd52w.parquet"   # must contain: date, qqq_close

# Execution assumptions (match your v1.1)
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0
COMMISSION_PER_TRADE_USD = 0.0

def entry_fill(open_px: float) -> float:
    return float(open_px) + (SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0)

def exit_fill(close_px: float) -> float:
    return float(close_px) - (SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0)

def net_ret_from_prices(entry_open, exit_close):
    e = entry_fill(entry_open)
    x = exit_fill(exit_close)
    r = (x / e) - 1.0
    # If you ever add commission, apply here (kept 0)
    if COMMISSION_PER_TRADE_USD != 0.0:
        r -= (2.0 * COMMISSION_PER_TRADE_USD) / e
    return r

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


def fwd_roll_min(s: pd.Series, window: int) -> pd.Series:
    # min over the NEXT `window` bars starting at the current bar
    a = s.copy()
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s: pd.Series, window: int) -> pd.Series:
    a = s.copy()
    return a.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]

def build_bo_events_for_ticker(g: pd.DataFrame, lookback: int, retest_k: int, band: float, hold_days: int) -> pd.DataFrame:
    """
    BO-1:
      - breakout day t when close[t] > prior_high[t] * (1+band)
      - within next retest_k days, find earliest day r where low[r] <= prior_high[t]*(1+band) AND close[r] >= prior_high[t]
      - enter at open[r+1], exit at close[r+1+hold_days]
    """
    g = g.sort_values("date").reset_index(drop=True).copy()

    # Prior N-day high excluding today
    prior_high = g["high"].shift(1).rolling(lookback, min_periods=lookback).max()

    # Breakout condition
    breakout = (g["close"] > prior_high * (1.0 + band)) & prior_high.notna()

    if not breakout.any():
        return pd.DataFrame()

    # Precompute forward min/max for MAE/MFE from entry day (entry uses next open, but we approximate excursion from entry day onward)
    # We'll compute min low / max high over hold_days starting at ENTRY DAY (which is r+1).
    fwd_min_low = fwd_roll_min(g["low"], hold_days)
    fwd_max_high = fwd_roll_max(g["high"], hold_days)

    rows = []
    b_idx = np.where(breakout.values)[0]

    for bi in b_idx:
        lvl = float(prior_high.iloc[bi])
        if not np.isfinite(lvl) or lvl <= 0:
            continue

        # Search retest window: days (bi+1 .. bi+retest_k)
        start = bi + 1
        end = min(len(g) - 1, bi + retest_k)
        if start > end:
            continue

        window = g.iloc[start:end+1]

        # Retest/hold condition
        cond = (window["low"] <= lvl * (1.0 + band)) & (window["close"] >= lvl)

        if not cond.any():
            continue

        # First retest day index in original g
        ri = int(cond.idxmax())  # idxmax returns first True index for boolean series
        # Entry is next day open
        entry_i = ri + 1
        exit_i = entry_i + hold_days

        if exit_i >= len(g):
            continue

        entry_open = float(g.loc[entry_i, "open"])
        exit_close = float(g.loc[exit_i, "close"])
        if not (np.isfinite(entry_open) and np.isfinite(exit_close)):
            continue

        # Net ret with slip
        net_ret = net_ret_from_prices(entry_open, exit_close)

        # MAE/MFE over hold window starting at ENTRY DAY (entry_i)
        # Use entry fill price for excursion baseline
        efill = entry_fill(entry_open)
        min_low = float(fwd_min_low.loc[entry_i])
        max_high = float(fwd_max_high.loc[entry_i])
        if np.isfinite(min_low) and np.isfinite(max_high) and efill > 0:
            mae = (min_low / efill) - 1.0
            mfe = (max_high / efill) - 1.0
        else:
            mae, mfe = np.nan, np.nan

        rows.append({
            "ticker": g.loc[entry_i, "ticker"],
            "breakout_date": g.loc[bi, "date"],
            "signal_date": g.loc[ri, "date"],
            "entry_date": g.loc[entry_i, "date"],
            "exit_date": g.loc[exit_i, "date"],
            "entry_open": entry_open,
            "exit_close": exit_close,
            "lookback": lookback,
            "retest_k": retest_k,
            "band": band,
            "hold_days": hold_days,
            "net_ret": net_ret,
            "mae": mae,
            "mfe": mfe,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def summarize(events: pd.DataFrame, n_min: int = 500) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    def p(x, q): 
        return np.quantile(x, q) if len(x) else np.nan

    g = events.groupby(["lookback","retest_k","band","hold_days"], as_index=False).agg(
        n=("net_ret","size"),
        win_rate=("net_ret", lambda x: float((x > 0).mean())),
        mean_ret=("net_ret","mean"),
        median_ret=("net_ret","median"),
        p25=("net_ret", lambda x: p(x, 0.25)),
        p75=("net_ret", lambda x: p(x, 0.75)),
        mean_mae=("mae","mean"),
        mean_mfe=("mfe","mean"),
    )
    g = g[g["n"] >= n_min].sort_values("mean_ret", ascending=False)
    return g

def year_focus(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    e = events.copy()
    e["year"] = pd.to_datetime(e["entry_date"]).dt.year
    y = e.groupby("year", as_index=False).agg(
        n=("net_ret","size"),
        win_rate=("net_ret", lambda x: float((x > 0).mean())),
        mean_ret=("net_ret","mean"),
        median_ret=("net_ret","median"),
        mean_mae=("mae","mean"),
        mean_mfe=("mfe","mean"),
    )
    return y.sort_values("year")

def main():
    lookbacks = [20, 50, 100]
    retest_ks = [3, 5, 10]
    bands = [0.0, 0.005]
    hold_days_list = [20]

    n_min = 500

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    needed = {"date","ticker","open","high","low","close"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in parquet: {missing}")

    qqq_risk_on, qqq_risk_on_strong = build_qqq_regime_maps(QQQ_PATH)

    all_events = []

    # Build events per config
    for lookback in lookbacks:
        for retest_k in retest_ks:
            for band in bands:
                for hold_days in hold_days_list:
                    evs = []
                    for _, g in df.groupby("ticker", sort=False):
                        e = build_bo_events_for_ticker(g, lookback, retest_k, band, hold_days)
                        if not e.empty:
                            evs.append(e)
                    ev = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame()
                    if ev.empty:
                        continue

                    # Add gated version (QQQ > SMA200 on signal_date)
                    ev["risk_on"] = ev["signal_date"].map(lambda d: qqq_risk_on.get(pd.Timestamp(d).normalize(), True))
                    ev["risk_on_strong"] = ev["signal_date"].map(lambda d: qqq_risk_on_strong.get(pd.Timestamp(d).normalize(), False))

                    ev["gated"] = False
                    all_events.append(ev)

    if not all_events:
        print("No BO events generated.")
        return

    events = pd.concat(all_events, ignore_index=True)

    # Ungated summary
    top_ungated = summarize(events, n_min=n_min)
    print("\nTop BO-1 configs by mean_ret (UNGATED, n>=500):")
    print(top_ungated.head(15).to_string(index=False))

    # Gated summary
    gated_events = events[events["risk_on"] == True].copy()
    top_gated = summarize(gated_events, n_min=n_min)
    print("\nTop BO-1 configs by mean_ret (WITH QQQ gate, n>=500):")
    print(top_gated.head(15).to_string(index=False))

    strong_events = events[events["risk_on_strong"] == True].copy()
    top_strong = summarize(strong_events, n_min=n_min)
    print("\nTop BO-1 configs by mean_ret (WITH STRONG gate: QQQ>SMA200 & SMA50>SMA200, n>=500):")
    print(top_strong.head(15).to_string(index=False))

    # Pick best config from ungated and show year focus (ungated vs gated)
    if not top_ungated.empty:
        best = top_ungated.iloc[0]
        mask = (
            (events["lookback"] == best["lookback"]) &
            (events["retest_k"] == best["retest_k"]) &
            (events["band"] == best["band"]) &
            (events["hold_days"] == best["hold_days"])
        )
        ev_best = events[mask].copy()
        ev_best_g = ev_best[ev_best["risk_on"] == True].copy()

        print("\nBO-1 year focus (best UNGATED config, net):")
        print(year_focus(ev_best).to_string(index=False))

        print("\nBO-1 year focus (best config WITH QQQ gate, net):")
        print(year_focus(ev_best_g).to_string(index=False))

        ev_best_s = ev_best[ev_best["risk_on_strong"] == True].copy()
        print("\nBO-1 year focus (best config WITH STRONG gate, net):")
        print(year_focus(ev_best_s).to_string(index=False))

       
if __name__ == "__main__":
    main()
