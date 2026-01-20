# panic_mr_sweep_fast.py
# Vectorized sweep for "Panic Mean Reversion" (MR-P)
#
# EOD signal at date t -> enter at open(t+1) -> exit at close(t+1+hold)
#
# Run:
#   python panic_mr_sweep_fast.py
#
import numpy as np
import pandas as pd
import itertools

STOCK_PATH = "data/sp100_daily_features.parquet"
QQQ_PATH   = "data/qqq_dd52w.parquet"

START_DATE = "2015-10-16"
END_DATE   = "2025-12-22"

MIN_N_ALL  = 150
MIN_N_2022 = 30

HOLDS = [3, 5, 7, 10]
RET1D_MAX = [-0.02, -0.03, -0.04, -0.05]
RET5D_MAX = [-0.06, -0.08, -0.10, -0.12]
ATR_MIN   = [0.02, 0.025, 0.03]
ALLOW_BELOW_SMA200 = [True, False]

# Stress modes:
# - risk_off: qqq trend broken (risk_on == False)
# - dd52w_leq: qqq dd52w <= threshold
DD52W_THRESHOLDS = [-0.10, -0.15, -0.20]

def load_data():
    print("Reading stock parquet...")
    df = pd.read_parquet(STOCK_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)
    df = df[(df["date"] >= pd.Timestamp(START_DATE)) & (df["date"] <= pd.Timestamp(END_DATE))].copy()

    print(f"Loaded stock rows: {len(df)} | tickers: {df['ticker'].nunique()} | dates: {df['date'].nunique()}")

    print("Reading QQQ parquet...")
    qqq = pd.read_parquet(QQQ_PATH).copy()
    qqq["date"] = pd.to_datetime(qqq["date"]).dt.normalize()
    qqq = qqq.sort_values("date").reset_index(drop=True)

    if "qqq_close" not in qqq.columns:
        raise RuntimeError(f"QQQ file missing 'qqq_close'. Columns: {qqq.columns.tolist()}")

    # build qqq features
    qqq["qqq_sma200"] = qqq["qqq_close"].rolling(200, min_periods=200).mean()
    qqq["risk_on"] = qqq["qqq_close"] > qqq["qqq_sma200"]

    if "qqq_dd52w" in qqq.columns:
        qqq["qqq_dd52w"] = qqq["qqq_dd52w"].astype(float)
    elif "dd_from_52w_high" in qqq.columns:
        qqq["qqq_dd52w"] = qqq["dd_from_52w_high"].astype(float)
    else:
        roll_high = qqq["qqq_close"].rolling(252, min_periods=252).max()
        qqq["qqq_dd52w"] = (qqq["qqq_close"] / roll_high) - 1.0

    qqq = qqq[["date","risk_on","qqq_dd52w"]]
    df = df.merge(qqq, on="date", how="left")

    df["year"] = df["date"].dt.year

    required = {"ticker","date","open","close","ret1d","ret5d","atr14_pct","risk_on","qqq_dd52w"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    return df

def main():
    df = load_data()

    # --- base arrays (vectorized) ---
    ret1d = df["ret1d"].to_numpy(dtype=float)
    ret5d = df["ret5d"].to_numpy(dtype=float)
    atr   = df["atr14_pct"].to_numpy(dtype=float)

    close = df["close"].to_numpy(dtype=float)
    sma200 = df["sma200"].to_numpy(dtype=float) if "sma200" in df.columns else None

    risk_on = df["risk_on"].fillna(True).to_numpy(dtype=bool)
    dd52w   = df["qqq_dd52w"].to_numpy(dtype=float)

    year = df["year"].to_numpy(dtype=int)

    # per-row validity (prevents tons of repeated isfinite checks)
    finite_base = np.isfinite(ret1d) & np.isfinite(ret5d) & np.isfinite(atr)

    # For allow_below=False we require close > sma200 and sma200 finite
    if sma200 is not None:
        above_sma200 = np.isfinite(sma200) & np.isfinite(close) & (close > sma200)
    else:
        above_sma200 = np.zeros(len(df), dtype=bool)

    # stress masks (vectorized, reused)
    stress_risk_off = ~risk_on
    stress_dd52w = {thr: (np.isfinite(dd52w) & (dd52w <= thr)) for thr in DD52W_THRESHOLDS}

    # precompute next-open entry prices once
    open_next = df.groupby("ticker")["open"].shift(-1).to_numpy(dtype=float)

    results = []
    base_configs = list(itertools.product(HOLDS, RET1D_MAX, RET5D_MAX, ATR_MIN, ALLOW_BELOW_SMA200))
    total_configs = (1 + len(DD52W_THRESHOLDS)) * len(base_configs)
    print(f"Starting sweep... Total configs: {total_configs}")

    cfg_i = 0
    for hold, r1max, r5max, amin, allow_below in base_configs:
        # correct exit close alignment (enter t+1 open, exit t+1+hold close)
        close_exit = df.groupby("ticker")["close"].shift(-(hold + 1)).to_numpy(dtype=float)
        fwd_ret = (close_exit / open_next) - 1.0
        fwd_ok = np.isfinite(fwd_ret) & np.isfinite(open_next) & (open_next > 0)

        # signal mask vectorized
        sig = (
            finite_base
            & (ret1d <= r1max)
            & (ret5d <= r5max)
            & (atr   >= amin)
        )
        if not allow_below:
            sig = sig & above_sma200

        base_mask = sig & fwd_ok

        # Evaluate both stress definitions for this base signal:
        # 1) risk_off
        for stress_mode, thr, stress_mask in [
            ("risk_off", np.nan, stress_risk_off),
            *[("dd52w_leq", float(t), stress_dd52w[t]) for t in DD52W_THRESHOLDS],
        ]:
            cfg_i += 1
            if cfg_i % 400 == 0:
                print(f"...config {cfg_i}/{total_configs}")

            mask = base_mask & stress_mask
            n_all = int(mask.sum())
            if n_all < MIN_N_ALL:
                continue

            mask_2022 = mask & (year == 2022)
            n_2022 = int(mask_2022.sum())
            if n_2022 < MIN_N_2022:
                continue

            rets_all = fwd_ret[mask]
            rets_22  = fwd_ret[mask_2022]

            results.append({
                "stress_mode": stress_mode,
                "stress_thr": thr,
                "hold": int(hold),
                "ret1d_max": float(r1max),
                "ret5d_max": float(r5max),
                "atr_min": float(amin),
                "allow_below_sma200": bool(allow_below),
                "n_all": n_all,
                "mean_all": float(np.mean(rets_all)),
                "win_all": float(np.mean(rets_all > 0)),
                "n_2022": n_2022,
                "mean_2022": float(np.mean(rets_22)),
                "win_2022": float(np.mean(rets_22 > 0)),
            })

    if not results:
        print("\nNo configs met thresholds. Lower MIN_N_ALL / MIN_N_2022 or broaden the grid.")
        return

    res = pd.DataFrame(results).sort_values("mean_2022", ascending=False).reset_index(drop=True)

    print("\n=== Top Panic MR configs (ranked by 2022 mean_ret) ===")
    print(res.head(25).to_string(index=False))

    print("\n=== Best config per stress_mode (by mean_2022) ===")
    for sm in ["risk_off", "dd52w_leq"]:
        sub = res[res["stress_mode"] == sm]
        if len(sub) == 0:
            print(f"\n {sm}\n  (no configs met thresholds)")
        else:
            print(f"\n {sm}")
            print(sub.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
