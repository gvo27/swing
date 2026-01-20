import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "bo1_year_focus.csv")

# Locked BO-1 v1
LOOKBACK = 100
RETEST_K = 5
BAND = 0.0
HOLD_DAYS = 20

ATR_MIN = 0.02
REQUIRE_ABOVE_SMA200 = True

SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0

def net_ret(entry_open: pd.Series, exit_close: pd.Series) -> pd.Series:
    slip = SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0
    entry_fill = entry_open.astype(float) + slip
    exit_fill = exit_close.astype(float) - slip
    return (exit_fill / entry_fill) - 1.0

def summarize(d: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(d),
        "win_rate": (d["net_ret"] > 0).mean() if len(d) else np.nan,
        "mean_ret": d["net_ret"].mean() if len(d) else np.nan,
        "median_ret": d["net_ret"].median() if len(d) else np.nan,
        "p25": d["net_ret"].quantile(0.25) if len(d) else np.nan,
        "p75": d["net_ret"].quantile(0.75) if len(d) else np.nan,
        "mean_mae": d["mae"].mean() if len(d) else np.nan,
        "mean_mfe": d["mfe"].mean() if len(d) else np.nan,
    })

def fwd_roll_min(s: pd.Series, window: int) -> pd.Series:
    a = s.shift(-1)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s: pd.Series, window: int) -> pd.Series:
    a = s.shift(-1)
    return a.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]

def build_events_for_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)

    level = g["close"].rolling(LOOKBACK, min_periods=LOOKBACK).max().shift(1)
    breakout = (g["close"] > level) & level.notna()

    g["entry_open_next"] = g["open"].shift(-1)
    g["exit_close_hold"] = g["close"].shift(-HOLD_DAYS)

    low_min = fwd_roll_min(g["low"], HOLD_DAYS)
    high_max = fwd_roll_max(g["high"], HOLD_DAYS)
    g["mae"] = (low_min / g["entry_open_next"]) - 1.0
    g["mfe"] = (high_max / g["entry_open_next"]) - 1.0

    rows = []
    idx_breaks = np.where(breakout.to_numpy(bool))[0]

    for t in idx_breaks:
        lvl = float(level.iloc[t])

        if not np.isfinite(g["atr14_pct"].iloc[t]) or g["atr14_pct"].iloc[t] < ATR_MIN:
            continue
        if REQUIRE_ABOVE_SMA200:
            if not np.isfinite(g["sma200"].iloc[t]) or g["close"].iloc[t] <= g["sma200"].iloc[t]:
                continue

        start = t + 1
        end = min(t + RETEST_K, len(g) - 1)

        found = None
        for r in range(start, end + 1):
            lo = float(g["low"].iloc[r])
            cl = float(g["close"].iloc[r])
            if lo <= lvl * (1.0 + BAND) and cl >= lvl:
                found = r
                break
        if found is None:
            continue

        r = found
        if r + 1 >= len(g) or r + HOLD_DAYS >= len(g):
            continue

        entry_open = g["entry_open_next"].iloc[r]
        exit_close = g["exit_close_hold"].iloc[r]
        mae = g["mae"].iloc[r]
        mfe = g["mfe"].iloc[r]

        if not (np.isfinite(entry_open) and np.isfinite(exit_close) and np.isfinite(mae) and np.isfinite(mfe)):
            continue

        rows.append({
            "signal_breakout_date": g["date"].iloc[t],
            "retest_date": g["date"].iloc[r],
            "entry_date": g["date"].iloc[r+1],
            "exit_date": g["date"].iloc[r+HOLD_DAYS],
            "year": int(pd.Timestamp(g["date"].iloc[r+1]).year),
            "entry_open": float(entry_open),
            "exit_close": float(exit_close),
            "mae": float(mae),
            "mfe": float(mfe),
        })

    if not rows:
        return pd.DataFrame()

    ev = pd.DataFrame(rows)
    ev["net_ret"] = net_ret(ev["entry_open"], ev["exit_close"])
    return ev

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    all_events = []
    for tkr, g in df.groupby("ticker", sort=False):
        ev = build_events_for_ticker(g)
        if not ev.empty:
            ev["ticker"] = tkr
            all_events.append(ev)

    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    if events.empty:
        raise RuntimeError("No BO-1 events produced.")

    events.to_csv(os.path.join(OUT_DIR, "bo1_events.csv"), index=False)

    by_year = events.groupby("year", as_index=False).apply(summarize).reset_index(drop=True)
    by_year.to_csv(OUT_CSV, index=False)

    focus_years = [2018, 2020, 2022, 2025]
    view = by_year[by_year["year"].isin(focus_years)].sort_values("year")
    print(f"Saved: {OUT_CSV}\n")
    print("BO-1 year focus (net):")
    print(view[["year","n","win_rate","mean_ret","median_ret","mean_mae","mean_mfe"]].to_string(index=False))

if __name__ == "__main__":
    main()
