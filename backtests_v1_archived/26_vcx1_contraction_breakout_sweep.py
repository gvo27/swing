import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "vcx1_contraction_breakout_sweep.csv")

# -----------------------------
# Slippage
# -----------------------------
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0
COMMISSION_PER_TRADE_USD = 0.0

def net_ret(entry_open: pd.Series, exit_close: pd.Series) -> pd.Series:
    slip = SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0
    entry_fill = entry_open.astype(float) + slip
    exit_fill = exit_close.astype(float) - slip
    r = (exit_fill / entry_fill) - 1.0
    if COMMISSION_PER_TRADE_USD != 0.0:
        r = r - (2.0 * COMMISSION_PER_TRADE_USD) / entry_fill
    return r

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

# -----------------------------
# VCX parameter grid
# -----------------------------
ATR_LOW_LIST = [0.015, 0.020, 0.025]
LOOKBACKS = [20, 50]
HOLDS = [10, 20]

REQUIRE_ABOVE_SMA200 = True

MIN_N = 500


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    needed = {"ticker","date","open","high","low","close","atr14_pct","sma200"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    rows = []

    for atr_low in ATR_LOW_LIST:
        for lb in LOOKBACKS:
            for hold in HOLDS:
                all_sigs = []

                for tkr, g in df.groupby("ticker", sort=False):
                    g = g.sort_values("date").reset_index(drop=True)
                    g = add_forward(g, hold=hold)

                    prior_high = g["close"].rolling(lb, min_periods=lb).max().shift(1)

                    cond = pd.Series(True, index=g.index)
                    cond &= g["atr14_pct"].notna() & (g["atr14_pct"] <= atr_low)

                    if REQUIRE_ABOVE_SMA200:
                        cond &= g["sma200"].notna() & (g["close"] > g["sma200"])

                    cond &= prior_high.notna()
                    cond &= (g["close"] > prior_high)

                    # execution availability
                    cond &= g["entry_open"].notna() & g["exit_close"].notna()
                    cond &= g["mae"].notna() & g["mfe"].notna()

                    sig = g.loc[cond, ["date","entry_open","exit_close","mae","mfe"]].copy()
                    if sig.empty:
                        continue

                    sig["ticker"] = tkr
                    sig["net_ret"] = net_ret(sig["entry_open"], sig["exit_close"])
                    all_sigs.append(sig)

                events = pd.concat(all_sigs, ignore_index=True) if all_sigs else pd.DataFrame()

                stats = summarize(events) if not events.empty else summarize(events)
                rows.append({
                    "atr_low": atr_low,
                    "lookback": lb,
                    "hold_days": hold,
                    **stats.to_dict()
                })

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}\n")

    filt = res[res["n"] >= MIN_N].copy()
    if filt.empty:
        print(f"No configs met MIN_N={MIN_N}.")
        print(res.sort_values("n", ascending=False).head(12).to_string(index=False))
        return

    print(f"Top VCX-1 configs by mean_ret (n>={MIN_N}):")
    print(filt.sort_values("mean_ret", ascending=False).head(15).to_string(index=False))

    print(f"\nTop VCX-1 configs by BEST mean_mae (n>={MIN_N}, less negative is better):")
    print(filt.sort_values("mean_mae", ascending=False).head(15).to_string(index=False))

if __name__ == "__main__":
    main()
