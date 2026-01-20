import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "tp_sweep_ret5d.csv")

# Holds to evaluate (entry next open, exit close at t+H)
HOLDS = [5, 10, 20]

# Sweep thresholds for ret5d (signal requires ret5d <= threshold)
THRESHOLDS = [-0.02, -0.03, -0.04, -0.05, -0.06, -0.08, -0.10]

# Trend filters (RSI-free)
REQUIRE_TREND = True         # above SMA200 and SMA200 slope > 0
REQUIRE_DOWN_DAY = True      # ret1d <= 0 on signal day


def fwd_roll_min(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]


def build_signal(g: pd.DataFrame, ret5d_max: float) -> pd.Series:
    cond = pd.Series(True, index=g.index)

    if REQUIRE_TREND:
        cond &= (g["above_sma200"] == 1)
        cond &= g["sma200_slope20"].notna()
        cond &= (g["sma200_slope20"] > 0)

    cond &= g["ret5d"].notna()
    cond &= (g["ret5d"] <= ret5d_max)

    if REQUIRE_DOWN_DAY:
        cond &= g["ret1d"].notna()
        cond &= (g["ret1d"] <= 0)

    # require entry day exists
    cond &= g["open"].shift(-1).notna()

    return cond


def add_forward_metrics(g: pd.DataFrame, holds: list[int]) -> pd.DataFrame:
    g = g.copy()
    g["entry_open"] = g["open"].shift(-1)

    for H in holds:
        g[f"exit_close_{H}d"] = g["close"].shift(-H)

        low_min = fwd_roll_min(g["low"], window=H, start_offset=1)
        high_max = fwd_roll_max(g["high"], window=H, start_offset=1)

        g[f"ret_{H}d"] = (g[f"exit_close_{H}d"] / g["entry_open"]) - 1.0
        g[f"mae_{H}d"] = (low_min / g["entry_open"]) - 1.0
        g[f"mfe_{H}d"] = (high_max / g["entry_open"]) - 1.0

    return g


def summarize(d: pd.DataFrame) -> dict:
    return {
        "n": int(len(d)),
        "win_rate": float((d["ret"] > 0).mean()) if len(d) else np.nan,
        "mean_ret": float(d["ret"].mean()) if len(d) else np.nan,
        "median_ret": float(d["ret"].median()) if len(d) else np.nan,
        "p25_ret": float(d["ret"].quantile(0.25)) if len(d) else np.nan,
        "p75_ret": float(d["ret"].quantile(0.75)) if len(d) else np.nan,
        "mean_mae": float(d["mae"].mean()) if len(d) else np.nan,
        "mean_mfe": float(d["mfe"].mean()) if len(d) else np.nan,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    needed = {"date","ticker","open","high","low","close","ret1d","ret5d","above_sma200","sma200_slope20"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in parquet: {missing}")

    rows = []

    # Precompute forward metrics once (independent of threshold)
    out_frames = []
    for tkr, g in df.groupby("ticker", sort=False):
        g = add_forward_metrics(g, HOLDS)
        out_frames.append(g)
    full = pd.concat(out_frames, ignore_index=True)

    for thr in THRESHOLDS:
        # Build signals per ticker (keeps logic clean)
        sig_frames = []
        for tkr, g in full.groupby("ticker", sort=False):
            g = g.copy()
            g["signal"] = build_signal(g, ret5d_max=thr).astype(int)
            sig_frames.append(g)
        all_sig = pd.concat(sig_frames, ignore_index=True)

        sig = all_sig[all_sig["signal"] == 1].copy()
        if sig.empty:
            for H in HOLDS:
                rows.append({"ret5d_max": thr, "hold_days": H, **summarize(pd.DataFrame(columns=["ret","mae","mfe"]))})
            continue

        for H in HOLDS:
            ev = sig[["date","ticker","entry_open", f"exit_close_{H}d", f"ret_{H}d", f"mae_{H}d", f"mfe_{H}d"]].copy()
            ev = ev.rename(columns={f"ret_{H}d":"ret", f"mae_{H}d":"mae", f"mfe_{H}d":"mfe"})
            ev = ev.dropna(subset=["entry_open","ret","mae","mfe"])
            rows.append({"ret5d_max": thr, "hold_days": H, **summarize(ev)})

    res = pd.DataFrame(rows).sort_values(["hold_days","ret5d_max"]).reset_index(drop=True)
    res.to_csv(OUT_CSV, index=False)

    print(f"Saved sweep: {OUT_CSV}")
    print("\nTop 3 by mean_ret (per hold):")
    for H in HOLDS:
        sub = res[res["hold_days"] == H].sort_values("mean_ret", ascending=False).head(3)
        print(f"\nHold {H}D:")
        print(sub[["ret5d_max","n","win_rate","mean_ret","median_ret","mean_mae","mean_mfe"]].to_string(index=False))


if __name__ == "__main__":
    main()
