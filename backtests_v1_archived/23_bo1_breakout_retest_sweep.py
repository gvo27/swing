import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "bo1_breakout_retest_sweep.csv")

# -----------------------------
# Slippage
# -----------------------------
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0
COMMISSION_PER_TRADE_USD = 0.0

def net_ret(entry_open: float, exit_close: float) -> float:
    slip = SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0
    entry_fill = float(entry_open) + slip
    exit_fill = float(exit_close) - slip
    r = (exit_fill / entry_fill) - 1.0
    if COMMISSION_PER_TRADE_USD != 0.0:
        r -= (2.0 * COMMISSION_PER_TRADE_USD) / entry_fill
    return r

# -----------------------------
# BO-1 parameter grid
# -----------------------------
LOOKBACKS = [20, 50, 100]         # breakout level window
RETEST_WINDOWS = [3, 5, 10]       # days after breakout to allow a retest entry
RETEST_BANDS = [0.0, 0.005]       # allow low <= level*(1+band)
HOLDS = [10, 20]                  # time exit

ATR_MIN = 0.02

# Light trend filter (start simple)
REQUIRE_ABOVE_SMA200 = True

MIN_N = 500


def fwd_roll_min(s: pd.Series, window: int) -> pd.Series:
    a = s.shift(-1)
    return a.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]

def fwd_roll_max(s: pd.Series, window: int) -> pd.Series:
    a = s.shift(-1)
    return a.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]

def add_forward(g: pd.DataFrame, hold: int) -> pd.DataFrame:
    g = g.copy()
    g["entry_open_next"] = g["open"].shift(-1)
    g["exit_close_hold"] = g["close"].shift(-hold)
    low_min = fwd_roll_min(g["low"], hold)
    high_max = fwd_roll_max(g["high"], hold)
    g["mae_hold"] = (low_min / g["entry_open_next"]) - 1.0
    g["mfe_hold"] = (high_max / g["entry_open_next"]) - 1.0
    return g

def summarize(d: pd.DataFrame) -> pd.Series:
    if len(d) == 0:
        return pd.Series({"n":0,"win_rate":np.nan,"mean_ret":np.nan,"median_ret":np.nan,"p25":np.nan,"p75":np.nan,"mean_mae":np.nan,"mean_mfe":np.nan})
    return pd.Series({
        "n": len(d),
        "win_rate": (d["net_ret"] > 0).mean(),
        "mean_ret": d["net_ret"].mean(),
        "median_ret": d["net_ret"].median(),
        "p25": d["net_ret"].quantile(0.25),
        "p75": d["net_ret"].quantile(0.75),
        "mean_mae": d["mae"].mean(),
        "mean_mfe": d["mfe"].mean(),
    })


def build_events_for_ticker(g: pd.DataFrame, lookback: int, retest_k: int, band: float, hold: int) -> pd.DataFrame:
    """
    For each breakout day t (close > prior N-day high), look for the FIRST retest day r in [t+1, t+retest_k]
    where low[r] <= level[t]*(1+band) and close[r] >= level[t].
    Enter at open[r+1], exit at close[r+hold].
    """
    g = g.copy().reset_index(drop=True)

    # breakout level based on prior closes
    level = g["close"].rolling(lookback, min_periods=lookback).max().shift(1)

    # breakout day condition
    breakout = (g["close"] > level) & level.notna()

    # forward fields for entry/exit for the RETEST day (r)
    # We'll use g's add_forward for the chosen hold (relative to the retest day index)
    g2 = add_forward(g, hold=hold)

    rows = []

    idx_breaks = np.where(breakout.to_numpy(bool))[0]
    if len(idx_breaks) == 0:
        return pd.DataFrame()

    for t in idx_breaks:
        lvl = float(level.iloc[t])

        # optional filters applied on breakout day (keep light)
        if g["atr14_pct"].iloc[t] < ATR_MIN:
            continue
        if REQUIRE_ABOVE_SMA200:
            if not np.isfinite(g["sma200"].iloc[t]) or g["close"].iloc[t] <= g["sma200"].iloc[t]:
                continue

        # search retest days
        start = t + 1
        end = min(t + retest_k, len(g) - 1)

        found = None
        for r in range(start, end + 1):
            lo = float(g["low"].iloc[r])
            cl = float(g["close"].iloc[r])

            if lo <= lvl * (1.0 + band) and cl >= lvl:
                found = r
                break

        if found is None:
            continue

        r = found

        # need entry open next day and exit close at r+hold
        if r + 1 >= len(g) or r + hold >= len(g):
            continue

        entry_open = g2["entry_open_next"].iloc[r]    # open[r+1]
        exit_close = g2["exit_close_hold"].iloc[r]    # close[r+hold]
        mae = g2["mae_hold"].iloc[r]
        mfe = g2["mfe_hold"].iloc[r]

        if not (np.isfinite(entry_open) and np.isfinite(exit_close) and np.isfinite(mae) and np.isfinite(mfe)):
            continue

        rows.append({
            "signal_breakout_date": g["date"].iloc[t],
            "retest_date": g["date"].iloc[r],
            "entry_date": g["date"].iloc[r+1],
            "exit_date": g["date"].iloc[r+hold],
            "entry_open": float(entry_open),
            "exit_close": float(exit_close),
            "net_ret": net_ret(entry_open, exit_close),
            "mae": float(mae),
            "mfe": float(mfe),
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    # sanity columns
    needed = {"ticker","date","open","high","low","close","atr14_pct","sma200"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    results = []

    for lookback in LOOKBACKS:
        for k in RETEST_WINDOWS:
            for band in RETEST_BANDS:
                for hold in HOLDS:
                    all_events = []
                    for tkr, g in df.groupby("ticker", sort=False):
                        ev = build_events_for_ticker(g, lookback=lookback, retest_k=k, band=band, hold=hold)
                        if not ev.empty:
                            ev["ticker"] = tkr
                            all_events.append(ev)

                    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()

                    if events.empty:
                        results.append({
                            "lookback": lookback, "retest_k": k, "band": band, "hold_days": hold,
                            "n": 0, "win_rate": np.nan, "mean_ret": np.nan, "median_ret": np.nan,
                            "p25": np.nan, "p75": np.nan, "mean_mae": np.nan, "mean_mfe": np.nan
                        })
                        continue

                    stats = summarize(events)
                    results.append({
                        "lookback": lookback,
                        "retest_k": k,
                        "band": band,
                        "hold_days": hold,
                        **stats.to_dict()
                    })

    res = pd.DataFrame(results)
    res.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}\n")

    filt = res[res["n"] >= MIN_N].copy()
    if filt.empty:
        print(f"No configs met MIN_N={MIN_N}. Try lowering MIN_N or expanding universe.")
        print(res.sort_values("n", ascending=False).head(15).to_string(index=False))
        return

    top = filt.sort_values("mean_ret", ascending=False).head(15)
    print(f"Top BO-1 configs by mean_ret (n>={MIN_N}):")
    print(top.to_string(index=False))

if __name__ == "__main__":
    main()
