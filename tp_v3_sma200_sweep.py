import os
import numpy as np
import pandas as pd

# =========================
# Config (TP_v2 baseline)
# =========================
PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "research_outputs"

HOLD_DAYS = 20
TP_RET5D_MAX = -0.04
TP_ATR_MIN = 0.02

# Entry/exit simulation for signal-quality study
USE_SLIPPAGE = True
SLIP_ABS = 0.02  # $0.02 per share each side

# Sweep for v3 filter
MIN_ABOVE_SMA200_SWEEP = [0.00, 0.02, 0.03, 0.05]

# If you want to focus on certain stress years:
FOCUS_YEARS = [2018, 2020, 2022, 2025]


# =========================
# Helpers
# =========================
def ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}\nHave: {sorted(df.columns.tolist())}")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def compute_forward_metrics(df: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    """
    Adds per-row forward metrics aligned to the SIGNAL date t:
      entry_open = open[t+1]
      exit_close = close[t+hold_days]
      min_low_h = min(low[t+1 ... t+hold_days])
      max_high_h = max(high[t+1 ... t+hold_days])
    """
    out = df.copy()

    # We'll compute per ticker to respect trading-day indexing
    def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()

        # Entry at next open
        g["entry_open"] = g["open"].shift(-1)

        # Exit at close after hold_days
        g["exit_close"] = g["close"].shift(-hold_days)

        # Forward min low / max high from t+1..t+hold_days
        low_next = g["low"].shift(-1)
        high_next = g["high"].shift(-1)

        g["min_low_h"] = low_next.rolling(hold_days, min_periods=hold_days).min().shift(-(hold_days - 1))
        g["max_high_h"] = high_next.rolling(hold_days, min_periods=hold_days).max().shift(-(hold_days - 1))

        return g

    out = out.groupby("ticker", group_keys=False).apply(_per_ticker)

    # Apply slippage (optional)
    if USE_SLIPPAGE:
        out["entry_px"] = out["entry_open"] + SLIP_ABS
        out["exit_px"] = out["exit_close"] - SLIP_ABS
    else:
        out["entry_px"] = out["entry_open"]
        out["exit_px"] = out["exit_close"]

    # Returns (signal date aligned)
    out["fwd_ret"] = (out["exit_px"] / out["entry_px"]) - 1.0

    # MAE/MFE relative to entry_px using forward min_low/max_high
    # (If slippage is used, we still measure excursions vs executed entry_px)
    out["mae"] = (out["min_low_h"] / out["entry_px"]) - 1.0
    out["mfe"] = (out["max_high_h"] / out["entry_px"]) - 1.0

    return out


def tp_signal_mask(df: pd.DataFrame, min_above_sma200: float) -> pd.Series:
    """
    TP_v3 = TP_v2 + (above_sma200 >= min_above_sma200)
    """
    cond = pd.Series(True, index=df.index)

    # Must have enough indicator history
    cond &= df["sma200"].notna()
    cond &= df["sma200_slope20"].notna()

    # Trend integrity
    cond &= df["sma200_slope20"] > 0
    cond &= df["close"] > df["sma200"]

    # New v3 filter
    cond &= df["above_sma200"].notna()
    cond &= df["above_sma200"] >= min_above_sma200

    # Pullback constraints
    cond &= df["ret5d"].notna()
    cond &= df["ret5d"] <= TP_RET5D_MAX  # e.g., <= -0.04
    cond &= df["ret1d"].notna()
    cond &= df["ret1d"] <= 0            # require down day

    # Volatility floor
    cond &= df["atr14_pct"].notna()
    cond &= df["atr14_pct"] >= TP_ATR_MIN

    return cond


def summarize_trades(trades: pd.DataFrame) -> dict:
    """
    trades must have fwd_ret, mae, mfe
    """
    r = trades["fwd_ret"].dropna()
    if len(r) == 0:
        return {
            "n": 0,
            "win_rate": np.nan,
            "mean_ret": np.nan,
            "median_ret": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "mean_mae": np.nan,
            "mean_mfe": np.nan,
        }

    return {
        "n": float(len(r)),
        "win_rate": float((r > 0).mean()),
        "mean_ret": float(r.mean()),
        "median_ret": float(r.median()),
        "p25": float(r.quantile(0.25)),
        "p75": float(r.quantile(0.75)),
        "mean_mae": float(trades.loc[r.index, "mae"].mean()),
        "mean_mfe": float(trades.loc[r.index, "mfe"].mean()),
    }


def year_table(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["year"] = pd.to_datetime(t["date"]).dt.year
    rows = []
    for y, g in t.groupby("year"):
        stats = summarize_trades(g)
        rows.append({"year": int(y), **stats})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_data(PARQUET_PATH)

    required = [
        "date", "ticker",
        "open", "high", "low", "close",
        "sma200", "sma200_slope20", "above_sma200",
        "ret1d", "ret5d",
        "atr14_pct",
    ]
    ensure_columns(df, required)

    # Add forward metrics (entry/exit/mae/mfe)
    full = compute_forward_metrics(df, HOLD_DAYS)

    # Need these to compute forward returns safely
    fwd_required = ["entry_px", "exit_px", "fwd_ret", "mae", "mfe"]
    ensure_columns(full, fwd_required)

    results = []
    year_tables = {}

    for x in MIN_ABOVE_SMA200_SWEEP:
        mask = tp_signal_mask(full, min_above_sma200=x)

        # Keep only rows where the forward return exists
        # (i.e. we have entry day and exit day available)
        mask &= full["entry_px"].notna()
        mask &= full["exit_px"].notna()
        mask &= full["fwd_ret"].notna()

        trades = full.loc[mask, ["date", "ticker", "fwd_ret", "mae", "mfe"]].copy()
        stats = summarize_trades(trades)

        results.append({
            "model": "TP_v3",
            "min_above_sma200": x,
            "hold_days": HOLD_DAYS,
            **stats
        })

        yt = year_table(trades)
        year_tables[x] = yt

        # Print focus years snapshot
        focus = yt[yt["year"].isin(FOCUS_YEARS)]
        if len(focus) > 0:
            print(f"\nTP_v3 | min_above_sma200={x:.2f} | hold={HOLD_DAYS}D")
            print(focus.to_string(index=False))

    res_df = pd.DataFrame(results).sort_values("min_above_sma200").reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, f"tp_v3_sma200_sweep_hold{HOLD_DAYS}.csv")
    res_df.to_csv(out_path, index=False)

    print("\n=== TP_v3 SMA200 Distance Sweep (overall) ===")
    print(res_df.to_string(index=False))
    print(f"\nSaved: {out_path}")

    # Also save per-year tables per threshold
    for x, yt in year_tables.items():
        p = os.path.join(OUT_DIR, f"tp_v3_years_minAbove_{x:.2f}_hold{HOLD_DAYS}.csv")
        yt.to_csv(p, index=False)

    print("\nSaved per-year tables to research_outputs/")

if __name__ == "__main__":
    main()
