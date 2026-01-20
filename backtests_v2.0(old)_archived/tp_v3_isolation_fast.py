import itertools
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
QQQ_PATH = "data/qqq_dd52w.parquet"

START_DATE = "2015-10-16"
END_DATE   = "2025-12-22"

SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0

def entry_fill(open_px: float) -> float:
    return float(open_px) + (SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0)

def exit_fill(close_px: float) -> float:
    return float(close_px) - (SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0)

def build_qqq_features(qqq_path: str) -> pd.DataFrame:
    q = pd.read_parquet(qqq_path).copy()
    q["date"] = pd.to_datetime(q["date"]).dt.normalize()
    if "qqq_close" not in q.columns:
        raise RuntimeError(f"QQQ file missing 'qqq_close'. Columns: {q.columns.tolist()}")
    q = q.sort_values("date").reset_index(drop=True)
    q["qqq_sma200"] = q["qqq_close"].rolling(200, min_periods=200).mean()
    q["risk_on"] = q["qqq_close"] > q["qqq_sma200"]
    q["qqq_ret20d"] = q["qqq_close"].pct_change(20)
    return q[["date","risk_on","qqq_ret20d"]]

def gate_mask(tr: pd.DataFrame, gate_mode: str, ret20d_min: float) -> pd.Series:
    if gate_mode == "none":
        return pd.Series(True, index=tr.index)
    if gate_mode == "risk_on":
        return tr["risk_on"].astype(bool)
    if gate_mode == "ret20d":
        return tr["qqq_ret20d"].notna() & (tr["qqq_ret20d"] >= ret20d_min)
    if gate_mode == "risk_on_or_ret20d":
        return tr["risk_on"].astype(bool) | (tr["qqq_ret20d"].notna() & (tr["qqq_ret20d"] >= ret20d_min))
    raise RuntimeError(f"Unknown gate_mode={gate_mode}")

def precompute_tp_trades(df: pd.DataFrame, qqq: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    """
    Precompute all per-ticker, non-overlapping trades using:
      signal day d close -> entry d+1 open -> exit d+hold close.
    We do NOT apply TP thresholds here; we only compute the trade returns and features used for filtering.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    df = df.merge(qqq, on="date", how="left")
    df["risk_on"] = df["risk_on"].fillna(True)

    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    # precompute above_sma200
    df["above_sma200"] = (df["close"] / df["sma200"]) - 1.0

    # per ticker next-day open and future close for exit
    g = df.groupby("ticker", sort=False)

    df["entry_open"] = g["open"].shift(-1)
    df["entry_date"] = g["date"].shift(-1)

    df["exit_close"] = g["close"].shift(-1 - hold_days)
    df["exit_date"]  = g["date"].shift(-1 - hold_days)

    # valid rows require entry/exit prices
    tr = df.dropna(subset=["entry_open","exit_close","entry_date","exit_date"]).copy()

    # compute net-ish returns with slippage model
    tr["fill_in"] = tr["entry_open"].map(entry_fill)
    tr["fill_out"] = tr["exit_close"].map(exit_fill)
    tr = tr[(tr["fill_in"] > 0) & (tr["fill_out"] > 0)].copy()
    tr["ret"] = (tr["fill_out"] / tr["fill_in"]) - 1.0

    # rename signal date for clarity
    tr = tr.rename(columns={"date": "signal_date"})
    tr["entry_year"] = pd.to_datetime(tr["entry_date"]).dt.year

    keep = [
        "ticker","signal_date","entry_date","exit_date","entry_year","ret",
        "risk_on","qqq_ret20d",
        "ret1d","ret5d","atr14_pct","above_sma200","sma200_slope20"
    ]
    return tr[keep].reset_index(drop=True)

def summarize(tr: pd.DataFrame) -> dict:
    if len(tr) == 0:
        return {"n": 0}
    return {
        "n": int(len(tr)),
        "win": float((tr["ret"] > 0).mean()),
        "mean": float(tr["ret"].mean()),
        "median": float(tr["ret"].median()),
    }

def main():
    print("Reading parquet...", flush=True)
    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[(df["date"] >= pd.Timestamp(START_DATE)) & (df["date"] <= pd.Timestamp(END_DATE))].copy()
    print("Read done.", flush=True)

    req = {"date","ticker","open","close","sma200","sma200_slope20","ret1d","ret5d","atr14_pct"}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns in stock parquet: {miss}")

    qqq = build_qqq_features(QQQ_PATH)

    hold = 20
    print(f"Precomputing trades (hold={hold})...", flush=True)
    trades = precompute_tp_trades(df, qqq, hold_days=hold)
    print("Trades precomputed:", len(trades), "flush=True")

    # -----------------------------
    # Sweep (vectorized filters only)
    # -----------------------------
    gate_modes = ["none","risk_on","risk_on_or_ret20d","ret20d"]
    ret20d_mins = [-0.02, 0.00, 0.02]

    ret5d_maxs = [-0.04, -0.05, -0.06]
    atr_mins = [0.02, 0.025, 0.03]
    min_aboves = [0.00, 0.02, 0.05]
    slope_mins = [0.00]  # extend later if desired

    tighten_flags = [False, True]
    ro_ret5d_maxs = [-0.06, -0.08]   # stricter in risk-off
    ro_atr_mins = [0.02, 0.03]
    ro_min_aboves = [0.00, 0.02]
    ro_slope_mins = [0.00]

    results = []
    total = 0

    for gate_mode, ret20d_min, ret5d_max, atr_min, min_above, slope_min, tighten in itertools.product(
        gate_modes, ret20d_mins, ret5d_maxs, atr_mins, min_aboves, slope_mins, tighten_flags
    ):
        if not tighten:
            ro_grid = [(ret5d_max, atr_min, min_above, slope_min)]
        else:
            ro_grid = list(itertools.product(ro_ret5d_maxs, ro_atr_mins, ro_min_aboves, ro_slope_mins))

        # base masks
        base = (
            trades["ret5d"].notna() & (trades["ret5d"] <= ret5d_max) &
            trades["ret1d"].notna() & (trades["ret1d"] <= 0) &
            trades["atr14_pct"].notna() & (trades["atr14_pct"] >= atr_min) &
            trades["above_sma200"].notna() & (trades["above_sma200"] >= min_above) &
            trades["sma200_slope20"].notna() & (trades["sma200_slope20"] > slope_min)
        )

        gate = gate_mask(trades, gate_mode, ret20d_min)

        for ro_ret5d_max, ro_atr_min, ro_min_above, ro_slope_min in ro_grid:
            total += 1

            if tighten:
                ro = (
                    trades["ret5d"].notna() & (trades["ret5d"] <= ro_ret5d_max) &
                    trades["ret1d"].notna() & (trades["ret1d"] <= 0) &
                    trades["atr14_pct"].notna() & (trades["atr14_pct"] >= ro_atr_min) &
                    trades["above_sma200"].notna() & (trades["above_sma200"] >= ro_min_above) &
                    trades["sma200_slope20"].notna() & (trades["sma200_slope20"] > ro_slope_min)
                )
                cond = gate & np.where(trades["risk_on"].astype(bool), base, ro)
            else:
                cond = gate & base

            tr = trades[cond].copy()

            # avoid tiny configs
            n_all = len(tr)
            if n_all < 1500:
                continue

            tr22 = tr[tr["entry_year"] == 2022]
            if len(tr22) < 100:
                continue

            s_all = summarize(tr)
            s_22 = summarize(tr22)

            results.append({
                "hold": hold,
                "gate_mode": gate_mode,
                "ret20d_min": ret20d_min,
                "ret5d_max": ret5d_max,
                "atr_min": atr_min,
                "min_above_sma200": min_above,
                "tighten_ro": tighten,
                "ro_ret5d_max": ro_ret5d_max,
                "ro_atr_min": ro_atr_min,
                "ro_min_above_sma200": ro_min_above,
                "n_all": s_all["n"],
                "mean_all": s_all["mean"],
                "win_all": s_all["win"],
                "n_2022": s_22["n"],
                "mean_2022": s_22["mean"],
                "win_2022": s_22["win"],
            })

    print("Configs evaluated:", total, "kept:", len(results), flush=True)

    if not results:
        print("No configs met thresholds. Lower n_all/n_2022 thresholds.")
        return

    res = pd.DataFrame(results)

    # de-duplicate just in case
    res = res.drop_duplicates().sort_values(["mean_2022","mean_all"], ascending=False).reset_index(drop=True)

    print("\n=== Top TP_v3 configs (ranked by 2022 mean_ret, next-open execution) ===")
    print(res.head(30).to_string(index=False))

if __name__ == "__main__":
    main()
