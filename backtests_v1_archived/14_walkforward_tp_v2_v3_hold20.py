import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "tp_walkforward_hold20.csv")

# -----------------------------
# Shared settings
# -----------------------------
H = 20

# Slippage (Robinhood style)
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0

# Walk-forward: 4y train, 1y test
TRAIN_YEARS = 4

# Minimum trades required in a test year to report (avoid tiny-n noise)
MIN_TEST_N = 100


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
    # TP_v2: trend + pullback + down day + atr floor (no dd filter)
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
    # TP_v3 = TP_v2 + dd_from_52w_high >= -0.10
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

    # Precompute forward metrics per ticker
    frames = []
    for tkr, g in df.groupby("ticker", sort=False):
        frames.append(add_forward(g))
    full = pd.concat(frames, ignore_index=True)

    years = sorted(full["year"].dropna().unique().tolist())
    # We need full 52w/200d warmup etc; but that's already embedded. We'll just use years present.
    min_year = min(years)
    max_year = max(years)

    rows = []

    # Test years start after enough train years
    for test_year in range(min_year + TRAIN_YEARS, max_year + 1):
        train_start = test_year - TRAIN_YEARS
        train_end = test_year - 1

        # Note: we are not "training parameters" here; WF is for robustness reporting.
        # If you later add tunable params, you would choose them on train and apply to test.

        test = full[full["year"] == test_year].copy()
        if test.empty:
            continue

        for model_name, sig_fn in [("TP_v2", signal_tp_v2), ("TP_v3", signal_tp_v3)]:
            mask = sig_fn(test)
            ev = test[mask].copy()
            ev["net_ret"] = net_ret(ev["entry_open"], ev["exit_close"])

            stats = summarize(ev)
            rows.append({
                "model": model_name,
                "train_years": f"{train_start}-{train_end}",
                "test_year": int(test_year),
                **stats
            })

    res = pd.DataFrame(rows)

    # Flag tiny-n test years
    res["enough_n"] = res["n"] >= MIN_TEST_N

    res.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}\n")

    # Print a compact view (test years, per model)
    show = res.sort_values(["test_year", "model"])
    print(show[["test_year","model","n","win_net","mean_net","median_net","p25_net","p75_net","mean_mae","mean_mfe"]].to_string(index=False))

    # Overall out-of-sample averages (only years with enough trades)
    print(f"\nOut-of-sample averages (test years with n>={MIN_TEST_N}):")
    sub = res[res["enough_n"]].copy()
    if sub.empty:
        print("No test years met MIN_TEST_N. Lower MIN_TEST_N or widen universe.")
        return

    agg = sub.groupby("model").agg(
        test_years=("test_year","count"),
        avg_n=("n","mean"),
        avg_mean_net=("mean_net","mean"),
        avg_win_net=("win_net","mean"),
        avg_mean_mae=("mean_mae","mean"),
        avg_mean_mfe=("mean_mfe","mean"),
    ).reset_index()
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
