# panic_mr_p1_years.py
# MR_P1 isolation: year breakdown + overall stats
#
# MR_P1 (locked):
#   stress: QQQ dd52w <= -0.15
#   signal: ret1d <= -0.05 AND ret5d <= -0.10 AND atr14_pct >= 0.03
#   hold: 5 (enter next open, exit close after hold)
#   allow_below_sma200: True
#
import numpy as np
import pandas as pd

STOCK_PATH = "data/sp100_daily_features.parquet"
QQQ_PATH   = "data/qqq_dd52w.parquet"

START_DATE = "2015-10-16"
END_DATE   = "2025-12-22"

# Locked MR_P1 params
HOLD = 5
RET1D_MAX = -0.05
RET5D_MAX = -0.10
ATR_MIN   = 0.03
STRESS_THR = -0.15
ALLOW_BELOW_SMA200 = True

def load_data():
    df = pd.read_parquet(STOCK_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)
    df = df[(df["date"] >= pd.Timestamp(START_DATE)) & (df["date"] <= pd.Timestamp(END_DATE))].copy()

    qqq = pd.read_parquet(QQQ_PATH).copy()
    qqq["date"] = pd.to_datetime(qqq["date"]).dt.normalize()
    qqq = qqq.sort_values("date").reset_index(drop=True)

    if "qqq_close" not in qqq.columns:
        raise RuntimeError(f"QQQ file missing 'qqq_close'. Columns: {qqq.columns.tolist()}")

    # prefer existing dd52w if present
    if "qqq_dd52w" in qqq.columns:
        qqq["qqq_dd52w"] = qqq["qqq_dd52w"].astype(float)
    elif "dd_from_52w_high" in qqq.columns:
        qqq["qqq_dd52w"] = qqq["dd_from_52w_high"].astype(float)
    else:
        roll_high = qqq["qqq_close"].rolling(252, min_periods=252).max()
        qqq["qqq_dd52w"] = (qqq["qqq_close"] / roll_high) - 1.0

    qqq = qqq[["date","qqq_dd52w"]]
    df = df.merge(qqq, on="date", how="left")

    df["year"] = df["date"].dt.year
    return df

def main():
    df = load_data()

    required = {"ticker","date","open","close","ret1d","ret5d","atr14_pct","qqq_dd52w","year"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    # vector arrays
    ret1d = df["ret1d"].to_numpy(float)
    ret5d = df["ret5d"].to_numpy(float)
    atr   = df["atr14_pct"].to_numpy(float)
    dd52w = df["qqq_dd52w"].to_numpy(float)
    year  = df["year"].to_numpy(int)

    finite_base = np.isfinite(ret1d) & np.isfinite(ret5d) & np.isfinite(atr) & np.isfinite(dd52w)

    # stress
    stress = finite_base & (dd52w <= STRESS_THR)

    # signal
    sig = (
        finite_base
        & (ret1d <= RET1D_MAX)
        & (ret5d <= RET5D_MAX)
        & (atr   >= ATR_MIN)
    )

    if not ALLOW_BELOW_SMA200:
        sma200 = df["sma200"].to_numpy(float)
        close  = df["close"].to_numpy(float)
        sig = sig & np.isfinite(sma200) & np.isfinite(close) & (close > sma200)

    mask = sig & stress

    # forward return: enter t+1 open, exit close at t+1+HOLD
    open_next = df.groupby("ticker")["open"].shift(-1).to_numpy(float)
    close_exit = df.groupby("ticker")["close"].shift(-(HOLD + 1)).to_numpy(float)
    fwd_ret = (close_exit / open_next) - 1.0
    ok = mask & np.isfinite(fwd_ret) & np.isfinite(open_next) & (open_next > 0)

    n_all = int(ok.sum())
    if n_all == 0:
        print("No trades found for MR_P1.")
        return

    rets = fwd_ret[ok]
    print("\n=== MR_P1 isolation (locked) ===")
    print(f"Stress: QQQ dd52w <= {STRESS_THR}")
    print(f"Signal: ret1d <= {RET1D_MAX}, ret5d <= {RET5D_MAX}, atr14_pct >= {ATR_MIN}")
    print(f"Hold: {HOLD} (next open -> close after hold)")
    print(f"Trades (all): n={n_all}, win={np.mean(rets>0):.3f}, mean={np.mean(rets):.4f}, median={np.median(rets):.4f}")

    # year breakdown
    years = sorted(df["year"].unique().tolist())
    rows = []
    for y in years:
        yy = ok & (year == y)
        n = int(yy.sum())
        if n == 0:
            continue
        rr = fwd_ret[yy]
        rows.append({
            "year": int(y),
            "n": n,
            "win_rate": float(np.mean(rr > 0)),
            "mean_ret": float(np.mean(rr)),
            "median_ret": float(np.median(rr)),
            "p25": float(np.quantile(rr, 0.25)),
            "p75": float(np.quantile(rr, 0.75)),
        })

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    print("\nYear breakdown:")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
