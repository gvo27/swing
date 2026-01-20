import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "tp_sweep_atr_filter.csv")

HOLDS = [5, 10, 20]

# Locked TP_v1 core signal (from your sweeps)
RET5D_MAX = -0.04
REQUIRE_TREND = True
REQUIRE_DOWN_DAY = True

# Sweep ATR% filters
# Interpretation: keep signals where atr14_pct is between [min_atr, max_atr]
ATR_MINS = [0.00, 0.01, 0.015, 0.02]
ATR_MAXS = [0.03, 0.04, 0.05, 0.07, 0.10]  # 0.10 is effectively "no cap" for most large caps


def fwd_roll_min(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]


def build_signal(g: pd.DataFrame, atr_min: float, atr_max: float) -> pd.Series:
    cond = pd.Series(True, index=g.index)

    if REQUIRE_TREND:
        cond &= (g["above_sma200"] == 1)
        cond &= g["sma200_slope20"].notna()
        cond &= (g["sma200_slope20"] > 0)

    cond &= g["ret5d"].notna()
    cond &= (g["ret5d"] <= RET5D_MAX)

    if REQUIRE_DOWN_DAY:
        cond &= g["ret1d"].notna()
        cond &= (g["ret1d"] <= 0)

    # ATR% filter (needs ATR computed)
    cond &= g["atr14_pct"].notna()
    cond &= (g["atr14_pct"] >= atr_min)
    cond &= (g["atr14_pct"] <= atr_max)

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

    needed = {"ticker","date","open","high","low","close","ret1d","ret5d","above_sma200","sma200_slope20","atr14_pct"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in parquet: {missing}")

    # Precompute forward metrics once
    frames = []
    for tkr, g in df.groupby("ticker", sort=False):
        frames.append(add_forward_metrics(g, HOLDS))
    full = pd.concat(frames, ignore_index=True)

    rows = []
    for atr_min in ATR_MINS:
        for atr_max in ATR_MAXS:
            if atr_max <= atr_min:
                continue

            sig_frames = []
            for tkr, g in full.groupby("ticker", sort=False):
                g = g.copy()
                g["signal"] = build_signal(g, atr_min=atr_min, atr_max=atr_max).astype(int)
                sig_frames.append(g)
            all_sig = pd.concat(sig_frames, ignore_index=True)

            sig = all_sig[all_sig["signal"] == 1].copy()

            for H in HOLDS:
                ev = sig[["date","ticker","entry_open", f"exit_close_{H}d", f"ret_{H}d", f"mae_{H}d", f"mfe_{H}d"]].copy()
                ev = ev.rename(columns={f"ret_{H}d":"ret", f"mae_{H}d":"mae", f"mfe_{H}d":"mfe"})
                ev = ev.dropna(subset=["entry_open","ret","mae","mfe"])
                rows.append({
                    "atr_min": atr_min,
                    "atr_max": atr_max,
                    "hold_days": H,
                    **summarize(ev)
                })

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")

    # Print "best" by mean_ret for each hold, with a minimum N guard
    MIN_N = 1000
    for H in HOLDS:
        sub = res[(res["hold_days"] == H) & (res["n"] >= MIN_N)].copy()
        sub = sub.sort_values("mean_ret", ascending=False).head(10)
        print(f"\nTop ATR filters by mean_ret (hold={H}D, n>={MIN_N}):")
        print(sub[["atr_min","atr_max","n","win_rate","mean_ret","median_ret","mean_mae","mean_mfe"]].to_string(index=False))

    # Also: best by MAE (risk control) for 20D
    sub20 = res[(res["hold_days"] == 20) & (res["n"] >= MIN_N)].copy()
    sub20 = sub20.sort_values("mean_mae", ascending=False).head(10)  # less negative is better
    print(f"\nTop ATR filters by BEST (least adverse) mean_mae (hold=20D, n>={MIN_N}):")
    print(sub20[["atr_min","atr_max","n","win_rate","mean_ret","median_ret","mean_mae","mean_mfe"]].to_string(index=False))


if __name__ == "__main__":
    main()
