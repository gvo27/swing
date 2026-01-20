import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "eb1_gap_sweep.csv")

# Slippage
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
# EB-1 sweep params
# -----------------------------
GAP_MINS = [0.03, 0.04, 0.05, 0.06]      # 3% to 6% gaps
HOLDS = [3, 5, 10]
ATR_MIN = 0.02

REQUIRE_ABOVE_SMA200 = True

MIN_N = 300

def build_signals(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["prev_close"] = g["close"].shift(1)
    g["gap"] = (g["open"] / g["prev_close"]) - 1.0
    g["day_ret"] = (g["close"] / g["open"]) - 1.0
    return g

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

    for gap_min in GAP_MINS:
        for hold in HOLDS:
            all_up = []
            all_dn = []

            for tkr, g in df.groupby("ticker", sort=False):
                g = g.sort_values("date").reset_index(drop=True)
                g = build_signals(g)
                g = add_forward(g, hold=hold)

                base = pd.Series(True, index=g.index)
                base &= g["gap"].notna() & g["day_ret"].notna()
                base &= g["atr14_pct"].notna() & (g["atr14_pct"] >= ATR_MIN)
                if REQUIRE_ABOVE_SMA200:
                    base &= g["sma200"].notna() & (g["close"] > g["sma200"])

                # execution availability
                base &= g["entry_open"].notna() & g["exit_close"].notna()
                base &= g["mae"].notna() & g["mfe"].notna()

                # EB_UP_CONT: gap up + green day
                cond_up = base & (g["gap"] >= gap_min) & (g["day_ret"] >= 0)
                up = g.loc[cond_up, ["date","entry_open","exit_close","mae","mfe"]].copy()
                if not up.empty:
                    up["ticker"] = tkr
                    up["net_ret"] = net_ret(up["entry_open"], up["exit_close"])
                    all_up.append(up)

                # EB_DN_REV: gap down + green day (reversal)
                cond_dn = base & (g["gap"] <= -gap_min) & (g["day_ret"] >= 0)
                dn = g.loc[cond_dn, ["date","entry_open","exit_close","mae","mfe"]].copy()
                if not dn.empty:
                    dn["ticker"] = tkr
                    dn["net_ret"] = net_ret(dn["entry_open"], dn["exit_close"])
                    all_dn.append(dn)

            up_events = pd.concat(all_up, ignore_index=True) if all_up else pd.DataFrame()
            dn_events = pd.concat(all_dn, ignore_index=True) if all_dn else pd.DataFrame()

            up_stats = summarize(up_events)
            dn_stats = summarize(dn_events)

            rows.append({
                "model": "EB_UP_CONT",
                "gap_min": gap_min,
                "hold_days": hold,
                **up_stats.to_dict()
            })
            rows.append({
                "model": "EB_DN_REV",
                "gap_min": gap_min,
                "hold_days": hold,
                **dn_stats.to_dict()
            })

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}\n")

    for model in ["EB_UP_CONT", "EB_DN_REV"]:
        filt = res[(res["model"] == model) & (res["n"] >= MIN_N)].copy()
        print(f"{model}: Top configs by mean_ret (n>={MIN_N})")
        if filt.empty:
            print("  (none met MIN_N)")
            continue
        print(filt.sort_values("mean_ret", ascending=False).head(12).to_string(index=False))
        print()

if __name__ == "__main__":
    main()
