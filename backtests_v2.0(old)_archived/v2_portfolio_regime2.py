import math
import numpy as np
import pandas as pd

# ----------------------------
# Paths
# ----------------------------
DATA_PATH = "data/sp100_daily_features.parquet"
REGIME_PATH = "data/qqq_regime.parquet"

# ----------------------------
# Strategy params (your locked ones)
# ----------------------------
TP_HOLD = 20
TP_RET5D_MAX = -0.04
TP_ATR_MIN = 0.02

MR_HOLD = 10
MR_RET3D_MAX = -0.07
MR_ATR_MIN = 0.02

# BO v2 candidate (locked)
BO_LOOKBACK = 100
BO_RETEST_K = 3
BO_BAND = 0.005
BO_HOLD = 20

# ----------------------------
# Portfolio controls (drawdown-first)
# ----------------------------
START_EQUITY = 100_000.0
MAX_TOTAL_POSITIONS = 30

EXPOSURE = {
    "STRONG": 1.00,
    "NEUTRAL": 0.60,
    "RISK_OFF": 0.25,
}

# Regime-specific caps (counts of concurrently OPEN positions)
CAPS = {
    "STRONG":   {"TP": 30, "MR": 6,  "BO": 4},   # BO steals from TP capacity (replacement)
    "NEUTRAL":  {"TP": 12, "MR": 10, "BO": 0},
    "RISK_OFF": {"TP": 0,  "MR": 6,  "BO": 0},
}

# Execution assumptions
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0

def entry_fill(open_px: float) -> float:
    return float(open_px) + (SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0)

def exit_fill(close_px: float) -> float:
    return float(close_px) - (SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0)

# ----------------------------
# Signals
# ----------------------------
def tp_signal(row) -> bool:
    # Minimal TP_v2 using your locked params.
    # If your v1.1 TP has extra filters, add them here for exact match.
    if not np.isfinite(row["ret5d"]):
        return False
    if row["ret5d"] > TP_RET5D_MAX:
        return False
    if not np.isfinite(row["atr14_pct"]):
        return False
    if row["atr14_pct"] < TP_ATR_MIN:
        return False
    return True

def mr_signal(row) -> bool:
    if not np.isfinite(row["ret3d"]):
        return False
    if row["ret3d"] > MR_RET3D_MAX:
        return False
    if not np.isfinite(row["sma10"]):
        return False
    if row["close"] >= row["sma10"]:
        return False
    if not np.isfinite(row["atr14_pct"]):
        return False
    if row["atr14_pct"] < MR_ATR_MIN:
        return False
    return True

# ----------------------------
# BO event builder (signal_date -> entry next day open)
# ----------------------------
def build_bo_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns events with columns:
      ticker, signal_date, entry_date, exit_date
    Signal day = retest day (within retest_k after breakout),
    Entry day  = next trading day,
    Exit day   = entry day + BO_HOLD trading days (calendar trading days of the dataset)
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    events = []

    for tkr, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True).copy()

        # prior high over lookback, excluding current day
        prior_high = g["high"].shift(1).rolling(BO_LOOKBACK, min_periods=BO_LOOKBACK).max()
        breakout = (g["close"] > prior_high * (1.0 + BO_BAND)) & prior_high.notna()

        if not breakout.any():
            continue

        b_idx = np.where(breakout.to_numpy())[0]
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

            # retest day index in original g
            ri = int(cond.idxmax())

            entry_i = ri + 1
            exit_i = entry_i + BO_HOLD
            if exit_i >= len(g):
                continue

            events.append({
                "ticker": tkr,
                "signal_date": g.loc[ri, "date"],
                "entry_date": g.loc[entry_i, "date"],
                "exit_date": g.loc[exit_i, "date"],
            })

    return pd.DataFrame(events)

# ----------------------------
# Metrics
# ----------------------------
def compute_perf(equity: pd.Series) -> dict:
    equity = equity.dropna()
    start = equity.index.min()
    end = equity.index.max()
    years = (end - start).days / 365.25
    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    max_dd = float(dd.min())

    daily_ret = equity.pct_change().fillna(0.0)
    avg_daily = float(daily_ret.mean())
    vol_daily = float(daily_ret.std())

    return {
        "start": start,
        "end": end,
        "years": years,
        "total_return": total_ret,
        "cagr": cagr,
        "max_dd": max_dd,
        "avg_daily": avg_daily,
        "vol_daily": vol_daily,
    }

def yearly_returns(equity: pd.Series) -> pd.Series:
    # year-by-year simple return
    eq = equity.resample("Y").last()
    return eq.pct_change().dropna()

# ----------------------------
# Main simulation
# ----------------------------
def main():
    df = pd.read_parquet(DATA_PATH).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    needed = {"date","ticker","open","high","low","close","atr14_pct","ret5d"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {DATA_PATH}: {missing}")

    # Derive MR features on the fly if absent
    if "ret3d" not in df.columns:
        df["ret3d"] = df.groupby("ticker")["close"].pct_change(3)
    if "sma10" not in df.columns:
        df["sma10"] = df.groupby("ticker")["close"].transform(lambda s: s.rolling(10, min_periods=10).mean())

    # Load regime and map date->regime (confirmed)
    r = pd.read_parquet(REGIME_PATH).copy()
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    if "regime" not in r.columns:
        raise RuntimeError(f"{REGIME_PATH} missing 'regime' column.")
    regime_map = dict(zip(r["date"], r["regime"]))

    # Calendar (trading days)
    cal = pd.Index(sorted(df["date"].dt.normalize().unique()))
    cal = pd.DatetimeIndex(cal)

    # Build BO events once; gate by regime==STRONG on signal day (confirmed)
    bo_events = build_bo_events(df)
    if not bo_events.empty:
        bo_events["signal_day"] = pd.to_datetime(bo_events["signal_date"]).dt.normalize()
        bo_events["regime"] = bo_events["signal_day"].map(lambda d: regime_map.get(d, "NEUTRAL"))
        bo_events = bo_events[bo_events["regime"] == "STRONG"].drop(columns=["signal_day","regime"]).reset_index(drop=True)

    # Fast lookup: per (date)->rows, and per (ticker,date)->prices
    df["d"] = df["date"].dt.normalize()
    by_day = {d: g for d, g in df.groupby("d", sort=False)}

    # price lookup maps
    # We'll store per ticker a Series indexed by date for open/close
    open_map = {}
    close_map = {}
    for tkr, g in df.groupby("ticker", sort=False):
        g = g.sort_values("d")
        open_map[tkr] = pd.Series(g["open"].to_numpy(), index=g["d"].to_numpy())
        close_map[tkr] = pd.Series(g["close"].to_numpy(), index=g["d"].to_numpy())

    # BO signals keyed by signal date -> list of (ticker, entry_date, exit_date)
    bo_by_signal_day = {}
    if not bo_events.empty:
        bo_events["signal_day"] = pd.to_datetime(bo_events["signal_date"]).dt.normalize()
        bo_events["entry_day"] = pd.to_datetime(bo_events["entry_date"]).dt.normalize()
        bo_events["exit_day"] = pd.to_datetime(bo_events["exit_date"]).dt.normalize()
        for d, g in bo_events.groupby("signal_day", sort=False):
            bo_by_signal_day[d] = g[["ticker","entry_day","exit_day"]].to_dict("records")

    # Portfolio state
    cash = START_EQUITY
    positions = []  # list of dicts: ticker,strat,shares,entry_day,exit_day,entry_price
    pending_entries = {}  # entry_day -> list of dicts {ticker,strat,exit_day}

    equity_curve = []
    eq_dates = []

    def count_open_by_strat():
        c = {"TP":0, "MR":0, "BO":0}
        for p in positions:
            c[p["strat"]] += 1
        return c

    def is_open_ticker(ticker: str) -> bool:
        return any(p["ticker"] == ticker for p in positions)

    # Main loop over days
    for i, day in enumerate(cal):
        # --- 1) Execute entries at OPEN ---
        if day in pending_entries:
            # Determine equity at start-of-day using yesterday close marks (approx: cash + last close)
            # We will size new positions using CURRENT equity marked at yesterday close (already stored last loop),
            # but for simplicity, use cash + mark-to-market at today's open? We'll use yesterday close mark if available,
            # else cash only.
            # We'll compute "eq_now" after we mark positions at today's close anyway;
            # for sizing at open, use last known equity:
            eq_now_for_sizing = (equity_curve[-1] if equity_curve else START_EQUITY)

            # Per-position target allocation (no leverage)
            reg_entry = regime_map.get(day, "NEUTRAL")
            caps_entry = CAPS.get(reg_entry, CAPS["NEUTRAL"])
            exp_target = EXPOSURE.get(reg_entry, 0.60)
            cap_total = min(MAX_TOTAL_POSITIONS, caps_entry["TP"] + caps_entry["MR"] + caps_entry["BO"])
            cap_total = max(1, cap_total)

            target_alloc = (eq_now_for_sizing * exp_target) / cap_total

            # process in the order they were queued
            orders = pending_entries.pop(day)
            for o in orders:
                if len(positions) >= MAX_TOTAL_POSITIONS:
                    break
                if is_open_ticker(o["ticker"]):
                    continue

                tkr = o["ticker"]
                if tkr not in open_map or day not in open_map[tkr].index:
                    continue
                opx = float(open_map[tkr].loc[day])
                if not np.isfinite(opx) or opx <= 0:
                    continue

                fill = entry_fill(opx)
                if fill <= 0:
                    continue

                # Buy shares using target_alloc, but not more than cash
                alloc = min(target_alloc, cash)
                if alloc < 100:  # tiny positions not worth it
                    continue

                shares = math.floor(alloc / fill)
                if shares <= 0:
                    continue

                cost = shares * fill
                if cost > cash + 1e-6:
                    continue

                cash -= cost
                positions.append({
                    "ticker": tkr,
                    "strat": o["strat"],        # "TP" / "MR" / "BO"
                    "shares": shares,
                    "entry_day": day,
                    "exit_day": o["exit_day"],
                    "entry_price": fill,
                })

        # --- 2) Mark-to-market at CLOSE and execute exits at CLOSE ---
        # Compute portfolio value at close
        port_value = cash
        for p in positions:
            tkr = p["ticker"]
            if tkr in close_map and day in close_map[tkr].index:
                cpx = float(close_map[tkr].loc[day])
                if np.isfinite(cpx) and cpx > 0:
                    port_value += p["shares"] * cpx

        # Execute exits (sell at close fill)
        still_open = []
        for p in positions:
            if day == p["exit_day"]:
                tkr = p["ticker"]
                if tkr in close_map and day in close_map[tkr].index:
                    cpx = float(close_map[tkr].loc[day])
                    if np.isfinite(cpx) and cpx > 0:
                        sell = exit_fill(cpx) * p["shares"]
                        cash += sell
                    else:
                        # if missing close, keep open (rare)
                        still_open.append(p)
                else:
                    still_open.append(p)
            else:
                still_open.append(p)
        positions = still_open

        # Record equity after close/exits
        # Recompute value after exits for accuracy
        port_value = cash
        for p in positions:
            tkr = p["ticker"]
            if tkr in close_map and day in close_map[tkr].index:
                cpx = float(close_map[tkr].loc[day])
                if np.isfinite(cpx) and cpx > 0:
                    port_value += p["shares"] * cpx

        equity_curve.append(port_value)
        eq_dates.append(day)

        # --- 3) Generate signals today to ENTER next day (if exists) ---
        if i >= len(cal) - 1:
            continue
        next_day = cal[i + 1]

        regime = regime_map.get(day, "NEUTRAL")
        caps = CAPS.get(regime, CAPS["NEUTRAL"])

        # Current open counts
        open_counts = count_open_by_strat()
        total_open = len(positions)

        # BO replacement rule: BO steals from TP capacity in STRONG
        tp_cap = caps["TP"]
        mr_cap = caps["MR"]
        bo_cap = caps["BO"]

        # Effective TP cap reduced by BO open
        if regime == "STRONG":
            tp_cap_eff = max(0, tp_cap - open_counts["BO"])
        else:
            tp_cap_eff = tp_cap

        tp_avail = max(0, tp_cap_eff - open_counts["TP"])
        mr_avail = max(0, mr_cap - open_counts["MR"])
        bo_avail = max(0, bo_cap - open_counts["BO"])

        total_avail = max(0, MAX_TOTAL_POSITIONS - total_open)

        if total_avail <= 0:
            continue

        # Prepare candidate lists from today's rows
        day_rows = by_day.get(day, None)
        if day_rows is None or day_rows.empty:
            continue

        # Filter out tickers already open
        open_tickers = set(p["ticker"] for p in positions)

        # --- BO candidates (from prebuilt events keyed by signal day) ---
        bo_cands = []
        if bo_avail > 0 and regime == "STRONG":
            for rec in bo_by_signal_day.get(day, []):
                tkr = rec["ticker"]
                if tkr in open_tickers:
                    continue
                # sanity: ensure entry_day matches next_day (it should)
                if rec["entry_day"] != next_day:
                    continue
                bo_cands.append(rec)

        # --- TP candidates ---
        tp_cands = []
        if tp_avail > 0 and caps["TP"] > 0:
            g = day_rows
            for _, row in g.iterrows():
                tkr = row["ticker"]
                if tkr in open_tickers:
                    continue
                if tp_signal(row):
                    # exit day based on ticker calendar (use global calendar index)
                    exit_idx = i + 1 + TP_HOLD
                    if exit_idx < len(cal):
                        tp_cands.append({
                            "ticker": tkr,
                            "exit_day": cal[exit_idx],
                            "score": float(row.get("atr14_pct", 0.0)),  # rank: higher ATR% first
                        })

        # --- MR candidates ---
        mr_cands = []
        if mr_avail > 0 and caps["MR"] > 0:
            g = day_rows
            for _, row in g.iterrows():
                tkr = row["ticker"]
                if tkr in open_tickers:
                    continue
                if mr_signal(row):
                    exit_idx = i + 1 + MR_HOLD
                    if exit_idx < len(cal):
                        mr_cands.append({
                            "ticker": tkr,
                            "exit_day": cal[exit_idx],
                            "score": float(row.get("ret3d", 0.0)),  # rank: most negative ret3d
                        })

        # Rank candidates
        # BO has no score here; just take in order.
        tp_cands.sort(key=lambda x: x["score"], reverse=True)  # high ATR%
        mr_cands.sort(key=lambda x: x["score"])               # most negative ret3d first

        # Decide how many to take, respecting total_avail and per-strat avail
        orders_next = []

        # In STRONG: take BO first (replacement). Each BO effectively reduces TP capacity already via tp_cap_eff.
        if regime == "STRONG" and bo_avail > 0 and total_avail > 0:
            take = min(bo_avail, total_avail, len(bo_cands))
            for k in range(take):
                orders_next.append({
                    "ticker": bo_cands[k]["ticker"],
                    "strat": "BO",
                    "exit_day": bo_cands[k]["exit_day"],
                })
            total_avail -= take

        # Then TP
        if tp_avail > 0 and total_avail > 0:
            take = min(tp_avail, total_avail, len(tp_cands))
            for k in range(take):
                orders_next.append({
                    "ticker": tp_cands[k]["ticker"],
                    "strat": "TP",
                    "exit_day": tp_cands[k]["exit_day"],
                })
            total_avail -= take

        # Then MR
        if mr_avail > 0 and total_avail > 0:
            take = min(mr_avail, total_avail, len(mr_cands))
            for k in range(take):
                orders_next.append({
                    "ticker": mr_cands[k]["ticker"],
                    "strat": "MR",
                    "exit_day": mr_cands[k]["exit_day"],
                })
            total_avail -= take

        if orders_next:
            pending_entries.setdefault(next_day, []).extend(orders_next)

    # ---- Results ----
    equity = pd.Series(equity_curve, index=pd.DatetimeIndex(eq_dates)).sort_index()
    stats = compute_perf(equity)
    yr = yearly_returns(equity)

    print(f"Start: {stats['start'].date()}   End: {stats['end'].date()}   Years: {stats['years']:.2f}")
    print(f"Total return: {stats['total_return']*100:.2f}%")
    print(f"CAGR: {stats['cagr']*100:.2f}%")
    print(f"Max Drawdown: {stats['max_dd']*100:.2f}%")
    print(f"Avg daily ret: {stats['avg_daily']:.5f}   Daily vol: {stats['vol_daily']:.5f}")

    print("\nYearly returns:")
    if len(yr) == 0:
        print("(none)")
    else:
        # show as decimals like your prior outputs
        print(yr.to_string())

if __name__ == "__main__":
    main()
