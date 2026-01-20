import itertools
import numpy as np
import pandas as pd

# -----------------------------
# Paths (use your FROZEN research parquet + QQQ file)
# -----------------------------
PARQUET_PATH = "data/sp100_daily_features.parquet"  # point to your frozen backtest parquet
QQQ_PATH = "data/qqq_dd52w.parquet"                 # must contain: date, qqq_close

START_DATE = "2015-10-16"
END_DATE   = "2025-12-22"

# -----------------------------
# Execution assumptions (match portfolio engine)
# -----------------------------
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0
COMMISSION_PER_TRADE_USD = 0.0  # keep 0 for now

def entry_fill(open_px: float) -> float:
    return float(open_px) + (SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0)

def exit_fill(close_px: float) -> float:
    return float(close_px) - (SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0)

# -----------------------------
# QQQ features for gating
# -----------------------------
def build_qqq_features(qqq_path: str) -> pd.DataFrame:
    q = pd.read_parquet(qqq_path).copy()
    q["date"] = pd.to_datetime(q["date"]).dt.normalize()
    if "qqq_close" not in q.columns:
        raise RuntimeError(f"QQQ file missing 'qqq_close'. Columns: {q.columns.tolist()}")

    q = q.sort_values("date").reset_index(drop=True)
    q["qqq_sma200"] = q["qqq_close"].rolling(200, min_periods=200).mean()
    q["risk_on"] = q["qqq_close"] > q["qqq_sma200"]
    q["qqq_ret20d"] = q["qqq_close"].pct_change(20)
    return q[["date", "risk_on", "qqq_ret20d"]]

def gate_ok(row, gate_mode: str, qqq_ret20d: float, ret20d_min: float) -> bool:
    """
    gate_mode:
      - "none": no gate (TP always allowed if signal true)
      - "risk_on": allow only if QQQ > SMA200
      - "ret20d": allow only if qqq_ret20d >= ret20d_min
      - "risk_on_or_ret20d": allow if risk_on OR qqq_ret20d >= ret20d_min
    """
    if gate_mode == "none":
        return True
    if gate_mode == "risk_on":
        return bool(row["risk_on"])
    if gate_mode == "ret20d":
        return np.isfinite(qqq_ret20d) and (qqq_ret20d >= ret20d_min)
    if gate_mode == "risk_on_or_ret20d":
        return bool(row["risk_on"]) or (np.isfinite(qqq_ret20d) and (qqq_ret20d >= ret20d_min))
    raise RuntimeError(f"Unknown gate_mode={gate_mode}")

# -----------------------------
# TP_v3 signal (TP_v2 base + stress tightening)
# -----------------------------
def tp_v3_signal(stock_row, *,
                 risk_on: bool,
                 ret5d_max: float,
                 atr_min: float,
                 min_above_sma200: float,
                 slope_min: float,
                 tighten_in_risk_off: bool,
                 ro_ret5d_max: float,
                 ro_atr_min: float,
                 ro_min_above_sma200: float,
                 ro_slope_min: float) -> bool:
    # basic feature availability
    if not (np.isfinite(stock_row.sma200) and np.isfinite(stock_row.sma200_slope20)):
        return False
    if not (np.isfinite(stock_row.ret5d) and np.isfinite(stock_row.ret1d) and np.isfinite(stock_row.atr14_pct)):
        return False
    if stock_row.close <= 0 or stock_row.sma200 <= 0:
        return False

    above = (stock_row.close / stock_row.sma200) - 1.0

    # choose thresholds (tighten when risk_off if enabled)
    if (not risk_on) and tighten_in_risk_off:
        _ret5d_max = ro_ret5d_max
        _atr_min = ro_atr_min
        _min_above = ro_min_above_sma200
        _slope_min = ro_slope_min
    else:
        _ret5d_max = ret5d_max
        _atr_min = atr_min
        _min_above = min_above_sma200
        _slope_min = slope_min

    # trend requirement
    if stock_row.sma200_slope20 <= _slope_min:
        return False
    if above < _min_above:
        return False

    # pullback requirement (same as TP_v2)
    if stock_row.ret5d > _ret5d_max:
        return False
    if stock_row.ret1d > 0:
        return False
    if stock_row.atr14_pct < _atr_min:
        return False

    return True

# -----------------------------
# Isolation backtest (one position per ticker, no overlap)
# -----------------------------
def run_tp_isolation(df: pd.DataFrame,
                     qqq_df: pd.DataFrame,
                     *,
                     hold_days: int,
                     gate_mode: str,
                     ret20d_min: float,
                     tighten_in_risk_off: bool,
                     ret5d_max: float,
                     atr_min: float,
                     min_above_sma200: float,
                     slope_min: float,
                     ro_ret5d_max: float,
                     ro_atr_min: float,
                     ro_min_above_sma200: float,
                     ro_slope_min: float) -> pd.DataFrame:
    # merge QQQ features by date
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    qqq_df = qqq_df.copy()
    qqq_df["date"] = pd.to_datetime(qqq_df["date"]).dt.normalize()

    df = df.merge(qqq_df, on="date", how="left")
    df["risk_on"] = df["risk_on"].fillna(True)

    # organize
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # map date -> next date per ticker existence (need open tomorrow)
    # We'll create trades only when ticker exists on entry day and exit day.
    trades = []

    # For "no overlap per ticker"
    next_free_date = {}  # ticker -> date (normalized)

    # Precompute dates index for exit lookup
    all_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    date_to_i = {pd.Timestamp(d): i for i, d in enumerate(all_dates)}

    # Group by date to access tomorrow open
    frames_by_date = {pd.Timestamp(d): g.set_index("ticker", drop=False) for d, g in df.groupby("date", sort=True)}

    for d in all_dates:
        day = pd.Timestamp(d)
        today = frames_by_date.get(day)
        if today is None or today.empty:
            continue

        # Need next trading day for entry
        di = date_to_i[day]
        if di >= len(all_dates) - 1:
            continue
        entry_day = pd.Timestamp(all_dates[di + 1])
        tomorrow = frames_by_date.get(entry_day)
        if tomorrow is None or tomorrow.empty:
            continue

        # Compute exit day based on entry_day index
        entry_i = date_to_i[entry_day]
        exit_i = entry_i + hold_days
        if exit_i >= len(all_dates):
            continue
        exit_day = pd.Timestamp(all_dates[exit_i])
        exit_frame = frames_by_date.get(exit_day)
        if exit_frame is None or exit_frame.empty:
            continue

        common = today.index.intersection(tomorrow.index)
        common = common.intersection(exit_frame.index)

        # per ticker signals at EOD today, enter next open
        for tkr in common:
            # no overlap per ticker
            nf = next_free_date.get(tkr, None)
            if nf is not None and day < nf:
                continue

            row = today.loc[tkr]
            qqq_ret20d = float(row.qqq_ret20d) if np.isfinite(row.qqq_ret20d) else np.nan

            if not gate_ok(row, gate_mode, qqq_ret20d, ret20d_min):
                continue

            ok = tp_v3_signal(
                row,
                risk_on=bool(row.risk_on),
                ret5d_max=ret5d_max,
                atr_min=atr_min,
                min_above_sma200=min_above_sma200,
                slope_min=slope_min,
                tighten_in_risk_off=tighten_in_risk_off,
                ro_ret5d_max=ro_ret5d_max,
                ro_atr_min=ro_atr_min,
                ro_min_above_sma200=ro_min_above_sma200,
                ro_slope_min=ro_slope_min,
            )
            if not ok:
                continue

            px_open = float(tomorrow.loc[tkr, "open"])
            px_close_exit = float(exit_frame.loc[tkr, "close"])
            fill_in = entry_fill(px_open)
            fill_out = exit_fill(px_close_exit)
            if not (np.isfinite(fill_in) and np.isfinite(fill_out) and fill_in > 0 and fill_out > 0):
                continue

            ret = (fill_out / fill_in) - 1.0

            trades.append({
                "signal_date": day,
                "entry_date": entry_day,
                "exit_date": exit_day,
                "ticker": str(tkr),
                "ret": float(ret),
                "risk_on": bool(row.risk_on),
                "qqq_ret20d": float(qqq_ret20d) if np.isfinite(qqq_ret20d) else np.nan,
            })

            # block overlaps until after exit (next signals allowed on/after exit_day)
            next_free_date[tkr] = exit_day

    return pd.DataFrame(trades)

def summarize(trades: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return {"n": 0}
    s = {}
    s["n"] = int(len(trades))
    s["win_rate"] = float((trades["ret"] > 0).mean())
    s["mean_ret"] = float(trades["ret"].mean())
    s["median_ret"] = float(trades["ret"].median())
    s["p25"] = float(trades["ret"].quantile(0.25))
    s["p75"] = float(trades["ret"].quantile(0.75))
    return s

def summarize_by_year(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=["year","n","win_rate","mean_ret","median_ret"])
    t = trades.copy()
    t["year"] = pd.to_datetime(t["entry_date"]).dt.year
    g = t.groupby("year")["ret"]
    out = pd.DataFrame({
        "n": g.size(),
        "win_rate": g.apply(lambda x: float((x > 0).mean())),
        "mean_ret": g.mean(),
        "median_ret": g.median(),
    }).reset_index()
    return out.sort_values("year").reset_index(drop=True)

def main():
    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    # lock research window
    df = df[(df["date"] >= pd.Timestamp(START_DATE)) & (df["date"] <= pd.Timestamp(END_DATE))].copy()

    # minimal required cols for TP + execution
    req = {"date","ticker","open","close","sma200","sma200_slope20","ret1d","ret5d","atr14_pct"}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns in stock parquet: {miss}")

    qqq = build_qqq_features(QQQ_PATH)

    # -----------------------------
    # Sweep space (kept intentionally small)
    # -----------------------------
    holds = [20]
    gate_modes = ["none", "risk_on"]
    ret20d_mins = [0.00]

    # base TP thresholds (normal regime)
    ret5d_maxs = [-0.04]
    atr_mins = [0.02]
    min_aboves = [0.00, 0.02]
    slope_mins = [0.00]  # keep 0.00 for now; add 0.0005 later if you want

    # risk-off tightening (only applies when tighten_in_risk_off=True)
    tighten_flags = [False]
    ro_ret5d_maxs = [-0.04, -0.06]   # risk-off: require deeper pullback
    ro_atr_mins = [0.02, 0.03]       # risk-off: require higher vol
    ro_min_aboves = [0.00, 0.02]     # risk-off: optionally demand above SMA200
    ro_slope_mins = [0.00]

    results = []

    print("Loaded stock rows:", len(df), "tickers:", df["ticker"].nunique(), "dates:", df["date"].nunique(), flush=True)
    print("Starting sweep...", flush=True)
    grid = list(itertools.product(
        holds,
        gate_modes, ret20d_mins,
        ret5d_maxs, atr_mins, min_aboves, slope_mins,
        tighten_flags,
        ro_ret5d_maxs, ro_atr_mins, ro_min_aboves, ro_slope_mins
    ))
    print("Total configs:", len(grid), flush=True)

    for i, (hold_days, gate_mode, ret20d_min,
            ret5d_max, atr_min, min_above, slope_min,
            tighten_in_risk_off,
            ro_ret5d_max, ro_atr_min, ro_min_above, ro_slope_min) in enumerate(grid, start=1):

        if i % 10 == 0:
            print(f"  ...config {i}/{len(grid)}", flush=True)

        # if not tightening, don't bother varying RO params (avoid duplicate work)
        if not tighten_in_risk_off:
            ro_ret5d_max = ret5d_max
            ro_atr_min = atr_min
            ro_min_above = min_above
            ro_slope_min = slope_min

        trades = run_tp_isolation(
            df, qqq,
            hold_days=hold_days,
            gate_mode=gate_mode,
            ret20d_min=ret20d_min,
            tighten_in_risk_off=tighten_in_risk_off,
            ret5d_max=ret5d_max,
            atr_min=atr_min,
            min_above_sma200=min_above,
            slope_min=slope_min,
            ro_ret5d_max=ro_ret5d_max,
            ro_atr_min=ro_atr_min,
            ro_min_above_sma200=ro_min_above,
            ro_slope_min=ro_slope_min,
        )

        if len(trades) == 0:
            continue

        by_year = summarize_by_year(trades)
        y2022 = by_year[by_year["year"] == 2022]
        n2022 = int(y2022["n"].iloc[0]) if len(y2022) else 0
        mean2022 = float(y2022["mean_ret"].iloc[0]) if len(y2022) else np.nan
        win2022 = float(y2022["win_rate"].iloc[0]) if len(y2022) else np.nan

        overall = summarize(trades)

        # thresholds to avoid tiny / misleading configs
        if overall["n"] < 1500:
            continue
        if n2022 < 100:
            continue

        results.append({
            "hold": hold_days,
            "gate_mode": gate_mode,
            "ret20d_min": ret20d_min,
            "ret5d_max": ret5d_max,
            "atr_min": atr_min,
            "min_above_sma200": min_above,
            "tighten_ro": tighten_in_risk_off,
            "ro_ret5d_max": ro_ret5d_max,
            "ro_atr_min": ro_atr_min,
            "ro_min_above_sma200": ro_min_above,
            "n_all": overall["n"],
            "mean_all": overall["mean_ret"],
            "win_all": overall["win_rate"],
            "n_2022": n2022,
            "mean_2022": mean2022,
            "win_2022": win2022,
        })

    if not results:
        print("No configs met thresholds. Try lowering n_all/n_2022 requirements.")
        return

    res = pd.DataFrame(results).sort_values(["mean_2022","mean_all"], ascending=False).reset_index(drop=True)

    print("\n=== Top TP_v3 configs (ranked by 2022 mean_ret, next-open execution) ===")
    print(res.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
