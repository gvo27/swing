import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "vcx1_year_focus.csv")

SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0
COMMISSION_PER_TRADE_USD = 0.0

# Two configs to compare
CONFIGS = [
    {"name": "VCX_A", "atr_low": 0.015, "lookback": 20, "hold_days": 20},
    {"name": "VCX_B", "atr_low": 0.015, "lookback": 20, "hold_days": 10},
]

REQUIRE_ABOVE_SMA200 = True

def net_ret(entry_open: pd.Series, exit_close: pd.Series) -> pd.Series:
    slip = SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0
    entry_fill = entry_open.astype(float) + slip
    exit_fill = exit_close.astype(float) - slip
    r = (exit_fill / entry_fill) - 1.0
    if COMMISSION_PER_TRADE_USD != 0.0:
        r = r - (2.0 * COMMISSION_PER_TRADE_USD) / entry_fill
    return r

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

    rows = []

    for cfg in CONFIGS:
        name = cfg["name"]
        atr_low = cfg["atr_low"]
        lb = cfg["lookback"]
        hold = cfg["hold_days"]

        all_events = []
        for tkr, g in df.groupby("ticker", sort=False):
            g = g.sort_values("date").reset_index(drop=True)
            g = add_forward(g, hold=hold)
            g["year"] = g["date"].dt.year

            prior_high = g["close"].rolling(lb, min_periods=lb).max().shift(1)

            cond = pd.Series(True, index=g.index)
            cond &= g["atr14_pct"].notna() & (g["atr14_pct"] <= atr_low)

            if REQUIRE_ABOVE_SMA200:
                cond &= g["sma200"].notna() & (g["close"] > g["sma200"])

            cond &= prior_high.notna()
            cond &= (g["close"] > prior_high)

            cond &= g["entry_open"].notna() & g["exit_close"].notna()
            cond &= g["mae"].notna() & g["mfe"].notna()

            sig = g.loc[cond, ["date","year","entry_open","exit_close","mae","mfe"]].copy()
            if sig.empty:
                continue

            sig["ticker"] = tkr
            sig["net_ret"] = net_ret(sig["entry_open"], sig["exit_close"])
            all_events.append(sig)

        events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
        if events.empty:
            continue

        by_year = events.groupby("year", as_index=False).apply(summarize).reset_index(drop=True)
        by_year["model"] = name
        by_year["atr_low"] = atr_low
        by_year["lookback"] = lb
        by_year["hold_days"] = hold
        rows.append(by_year)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}\n")

    focus_years = [2018, 2020, 2022, 2025]
    view = out[out["year"].isin(focus_years)].sort_values(["model","year"])
    print("VCX year focus (net):")
    print(view[["model","year","n","win_rate","mean_ret","median_ret","mean_mae","mean_mfe"]].to_string(index=False))

if __name__ == "__main__":
    main()
