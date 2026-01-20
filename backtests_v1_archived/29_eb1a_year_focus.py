import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "eb1a_year_focus.csv")

# EB-1A locked
GAP_MIN = 0.04
HOLD_DAYS = 10
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
        "mean_mae": d["mae"].mean() if len(d) else np.nan,
        "mean_mfe": d["mfe"].mean() if len(d) else np.nan,
    })

def fwd_roll_min(s: pd.Series, window: int) -> pd.Series:
    a = s.shift(-1)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s: pd.Series, window: int) -> pd.Series:
    a = s.shift(-1)
    return a.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]

def add_forward(g: pd.DataFrame, hold: int) -> pd.DataFrame:
    g = g.copy()
    g["entry_open"] = g["open"].shift(-1)
    g["exit_close"] = g["close"].shift(-hold)
    low_min = fwd_roll_min(g["low"], hold)
    high_max = fwd_roll_max(g["high"], hold)
    g["mae"] = (low_min / g["entry_open"]) - 1.0
    g["mfe"] = (high_max / g["entry_open"]) - 1.0
    return g

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    all_events = []
    for tkr, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        g["prev_close"] = g["close"].shift(1)
        g["gap"] = (g["open"] / g["prev_close"]) - 1.0
        g["day_ret"] = (g["close"] / g["open"]) - 1.0
        g["year"] = g["date"].dt.year
        g = add_forward(g, hold=HOLD_DAYS)

        cond = pd.Series(True, index=g.index)
        cond &= g["gap"].notna() & g["day_ret"].notna()
        cond &= (g["gap"] >= GAP_MIN)
        cond &= (g["day_ret"] >= 0)
        cond &= g["atr14_pct"].notna() & (g["atr14_pct"] >= ATR_MIN)

        if REQUIRE_ABOVE_SMA200:
            cond &= g["sma200"].notna() & (g["close"] > g["sma200"])

        cond &= g["entry_open"].notna() & g["exit_close"].notna()
        cond &= g["mae"].notna() & g["mfe"].notna()

        ev = g.loc[cond, ["date","year","entry_open","exit_close","mae","mfe"]].copy()
        if ev.empty:
            continue

        ev["ticker"] = tkr
        ev["net_ret"] = net_ret(ev["entry_open"], ev["exit_close"])
        all_events.append(ev)

    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    if events.empty:
        raise RuntimeError("No EB-1A events produced.")

    by_year = events.groupby("year", as_index=False).apply(summarize).reset_index(drop=True)
    by_year.to_csv(OUT_CSV, index=False)

    focus_years = [2018, 2020, 2022, 2025]
    view = by_year[by_year["year"].isin(focus_years)].sort_values("year")

    print(f"Saved: {OUT_CSV}\n")
    print("EB-1A year focus (net):")
    print(view[["year","n","win_rate","mean_ret","median_ret","mean_mae","mean_mfe"]].to_string(index=False))

if __name__ == "__main__":
    main()
