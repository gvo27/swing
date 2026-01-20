import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "tp_sweep_cooldown.csv")

HOLDS = [5, 10, 20]

# Locked TP_v1 params from your sweeps:
RET5D_MAX = -0.04
REQUIRE_TREND = True
REQUIRE_DOWN_DAY = True

COOLDOWNS = [0, 5, 10, 20]


def fwd_roll_min(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s: pd.Series, window: int, start_offset: int = 1) -> pd.Series:
    a = s.shift(-start_offset)
    return a.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]


def build_raw_signal(g: pd.DataFrame) -> pd.Series:
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

    # need entry day
    cond &= g["open"].shift(-1).notna()
    return cond


def apply_cooldown(raw_sig: pd.Series, cooldown_days: int) -> pd.Series:
    """
    raw_sig: boolean series indexed by time order for ONE ticker.
    cooldown_days: number of trading days to suppress after an accepted signal.
    """
    if cooldown_days <= 0:
        return raw_sig.astype(bool)

    accepted = np.zeros(len(raw_sig), dtype=bool)
    cooldown = 0
    raw = raw_sig.to_numpy(dtype=bool)

    for i in range(len(raw)):
        if cooldown > 0:
            cooldown -= 1
            continue
        if raw[i]:
            accepted[i] = True
            cooldown = cooldown_days  # suppress next N bars
    return pd.Series(accepted, index=raw_sig.index)


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

    # Precompute forward metrics once
    frames = []
    for tkr, g in df.groupby("ticker", sort=False):
        frames.append(add_forward_metrics(g, HOLDS))
    full = pd.concat(frames, ignore_index=True)

    rows = []
    for cd in COOLDOWNS:
        # Apply signals + cooldown per ticker
        sig_frames = []
        for tkr, g in full.groupby("ticker", sort=False):
            g = g.copy()
            raw = build_raw_signal(g)
            g["signal"] = apply_cooldown(raw, cooldown_days=cd).astype(int)
            sig_frames.append(g)

        all_sig = pd.concat(sig_frames, ignore_index=True)
        sig = all_sig[all_sig["signal"] == 1].copy()

        for H in HOLDS:
            ev = sig[["date","ticker","entry_open", f"exit_close_{H}d", f"ret_{H}d", f"mae_{H}d", f"mfe_{H}d"]].copy()
            ev = ev.rename(columns={f"ret_{H}d":"ret", f"mae_{H}d":"mae", f"mfe_{H}d":"mfe"})
            ev = ev.dropna(subset=["entry_open","ret","mae","mfe"])
            rows.append({
                "cooldown_days": cd,
                "ret5d_max": RET5D_MAX,
                "hold_days": H,
                **summarize(ev)
            })

    res = pd.DataFrame(rows).sort_values(["hold_days","cooldown_days"]).reset_index(drop=True)
    res.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")
    print(res[["cooldown_days","hold_days","n","win_rate","mean_ret","median_ret","mean_mae","mean_mfe"]].to_string(index=False))


if __name__ == "__main__":
    main()
