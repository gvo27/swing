import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
QQQ_DD_PATH = "data/qqq_dd52w.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "tp_walkforward_blend_hold20.csv")

H = 20
TRAIN_YEARS = 4
MIN_TEST_N = 100

# Slippage
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0

# --- Blend rule: if QQQ drawdown <= threshold -> risk-off -> TP_v3, else TP_v2
# Sweep a few sensible thresholds (you can add more)
BLEND_THRESHOLDS = [-0.05, -0.10, -0.15, -0.20]

def fwd_roll_min(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]

def add_forward(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["entry_open"] = g["open"].shift(-1)
    g["exit_close"] = g["close"].shift(-H)
    low_min = fwd_roll_min(g["low"], window=H, start_offset=1)
    high_max = fwd_roll_max(g["high"], window=H, start_offset=1)
    g["mae"] = (low_min / g["entry_open"]) - 1.0
    g["mfe"] = (high_max / g["entry_open"]) - 1.0
    return g

def net_ret(entry_open: pd.Series, exit_close: pd.Series) -> pd.Series:
    entry = entry_open.astype(float)
    exit_ = exit_close.astype(float)
    slip = SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0
    entry_fill = entry + slip
    exit_fill = exit_ - slip
    return (exit_fill / entry_fill) - 1.0

def signal_tp_v2(df: pd.DataFrame) -> pd.Series:
    cond = pd.Series(True, index=df.index)
    cond &= df["sma200"].notna()
    cond &= df["sma200_slope20"].notna()
    cond &= (df["sma200_slope20"] > 0)
    cond &= (df["close"] > df["sma200"])
    cond &= df["ret5d"].notna()
    cond &= (df["ret5d"] <= -0.04)
    cond &= df["ret1d"].notna()
    cond &= (df["ret1d"] <= 0)
    cond &= df["atr14_pct"].notna()
    cond &= (df["atr14_pct"] >= 0.02)
    cond &= df["entry_open"].notna()
    cond &= df["exit_close"].notna()
    cond &= df["mae"].notna()
    cond &= df["mfe"].notna()
    return cond

def signal_tp_v3(df: pd.DataFrame) -> pd.Series:
    cond = signal_tp_v2(df)
    cond &= df["dd_from_52w_high"].notna()
    cond &= (df["dd_from_52w_high"] >= -0.10)
    return cond

def summarize(d: pd.DataFrame) -> dict:
    return {
        "n": int(len(d)),
        "win_net": float((d["net_ret"] > 0).mean()) if len(d) else np.nan,
        "mean_net": float(d["net_ret"].mean()) if len(d) else np.nan,
        "median_net": float(d["net_ret"].median()) if len(d) else np.nan,
        "p25_net": float(d["net_ret"].quantile(0.25)) if len(d) else np.nan,
        "p75_net": float(d["net_ret"].quantile(0.75)) if len(d) else np.nan,
        "mean_mae": float(d["mae"].mean()) if len(d) else np.nan,
        "mean_mfe": float(d["mfe"].mean()) if len(d) else np.nan,
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year

    qqq = pd.read_parquet(QQQ_DD_PATH)
    qqq["date"] = pd.to_datetime(qqq["date"])

    # Merge QQQ dd by date
    df = df.merge(qqq[["date", "qqq_dd_from_52w_high"]], on="date", how="left")

    # Precompute forward metrics per ticker
    frames = []
    for tkr, g in df.groupby("ticker", sort=False):
        frames.append(add_forward(g))
    full = pd.concat(frames, ignore_index=True)

    years = sorted(full["year"].dropna().unique().tolist())
    min_year, max_year = min(years), max(years)

    rows = []

    for test_year in range(min_year + TRAIN_YEARS, max_year + 1):
        test = full[full["year"] == test_year].copy()
        if test.empty:
            continue

        # TP_v2
        ev2 = test[signal_tp_v2(test)].copy()
        ev2["net_ret"] = net_ret(ev2["entry_open"], ev2["exit_close"])
        rows.append({"model": "TP_v2", "blend_dd": np.nan, "test_year": test_year, **summarize(ev2)})

        # TP_v3
        ev3 = test[signal_tp_v3(test)].copy()
        ev3["net_ret"] = net_ret(ev3["entry_open"], ev3["exit_close"])
        rows.append({"model": "TP_v3", "blend_dd": np.nan, "test_year": test_year, **summarize(ev3)})

        # BLENDS
        for thr in BLEND_THRESHOLDS:
            risk_off = test["qqq_dd_from_52w_high"].notna() & (test["qqq_dd_from_52w_high"] <= thr)

            # Apply TP_v3 signals on risk-off days; TP_v2 on risk-on days
            m2 = signal_tp_v2(test) & (~risk_off)
            m3 = signal_tp_v3(test) & (risk_off)
            evb = test[m2 | m3].copy()
            evb["net_ret"] = net_ret(evb["entry_open"], evb["exit_close"])

            rows.append({"model": "BLEND", "blend_dd": thr, "test_year": test_year, **summarize(evb)})

    res = pd.DataFrame(rows)
    res["enough_n"] = res["n"] >= MIN_TEST_N
    res.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}\n")

    # Print per-year for blends (only the best blend threshold by mean_net each year)
    blends = res[res["model"] == "BLEND"].copy()
    if not blends.empty:
        best_each_year = blends.sort_values(["test_year", "mean_net"], ascending=[True, False]).groupby("test_year").head(1)
        print("Best BLEND threshold per test year (by mean_net):")
        print(best_each_year[["test_year","blend_dd","n","win_net","mean_net","median_net","mean_mae"]].to_string(index=False))

    # Overall out-of-sample averages (years with enough_n)
    print(f"\nOut-of-sample averages (test years with n>={MIN_TEST_N}):")
    sub = res[res["enough_n"]].copy()
    if sub.empty:
        print("No rows met MIN_TEST_N. Lower MIN_TEST_N.")
        return

    # For BLEND, show each threshold separately
    def agg_block(d: pd.DataFrame) -> pd.DataFrame:
        return d.groupby(["model","blend_dd"]).agg(
            test_years=("test_year","count"),
            avg_n=("n","mean"),
            avg_mean_net=("mean_net","mean"),
            avg_win_net=("win_net","mean"),
            avg_mean_mae=("mean_mae","mean"),
        ).reset_index()

    agg = agg_block(sub)
    # Sort: TP_v2/TP_v3 first, then blends by avg_mean_net desc
    agg["sort_key"] = agg["model"].map({"TP_v2":0,"TP_v3":1,"BLEND":2}).fillna(9).astype(int)
    agg = agg.sort_values(["sort_key","avg_mean_net"], ascending=[True, False]).drop(columns=["sort_key"])
    print(agg.to_string(index=False))

if __name__ == "__main__":
    main()
