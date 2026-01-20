import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "tp_v2_exit_sweep_hold20.csv")

# ---------- TP_v2 ENTRY (locked) ----------
RET5D_MAX = -0.04
ATR_MIN = 0.02
REQUIRE_DOWN_DAY = True
REQUIRE_TREND = True

HOLD_DAYS = 20  # exit window (consistent with your earlier shift(-20) convention)

# ---------- Slippage ----------
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0  # $0.02/share each side
COMMISSION_PER_TRADE_USD = 0.0

# ---------- Exit sweep grids ----------
# 0.0 means "disabled"
SL_K_LIST = [0.0, 1.0, 1.5, 2.0, 2.5]
TP_K_LIST = [0.0, 1.0, 1.5, 2.0, 3.0]

# Minimum trades per config to print (keeps noise down)
MIN_N = 500


def summarize(ret: np.ndarray, mae: np.ndarray, mfe: np.ndarray) -> dict:
    if len(ret) == 0:
        return dict(n=0, win_rate=np.nan, mean_ret=np.nan, median_ret=np.nan,
                    p25=np.nan, p75=np.nan, mean_mae=np.nan, mean_mfe=np.nan)
    s = pd.Series(ret)
    return dict(
        n=int(len(ret)),
        win_rate=float((ret > 0).mean()),
        mean_ret=float(s.mean()),
        median_ret=float(s.median()),
        p25=float(s.quantile(0.25)),
        p75=float(s.quantile(0.75)),
        mean_mae=float(np.mean(mae)),
        mean_mfe=float(np.mean(mfe)),
    )


def build_tp_v2_signal(g: pd.DataFrame) -> pd.Series:
    cond = pd.Series(True, index=g.index)

    if REQUIRE_TREND:
        cond &= g["sma200"].notna()
        cond &= g["sma200_slope20"].notna()
        cond &= (g["sma200_slope20"] > 0)
        cond &= (g["close"] > g["sma200"])

    cond &= g["ret5d"].notna()
    cond &= (g["ret5d"] <= RET5D_MAX)

    if REQUIRE_DOWN_DAY:
        cond &= g["ret1d"].notna()
        cond &= (g["ret1d"] <= 0)

    cond &= g["atr14_pct"].notna()
    cond &= (g["atr14_pct"] >= ATR_MIN)

    # Need next day's open for entry and t+H close for time exit availability
    cond &= g["open"].shift(-1).notna()
    cond &= g["close"].shift(-HOLD_DAYS).notna()
    return cond


def simulate_one(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_pct: np.ndarray,
    sig_i: int,
    sl_k: float,
    tp_k: float,
) -> tuple[float, float, float]:
    """
    Signal at index sig_i (day t).
    Entry at t+1 open.
    Time exit at close[t+HOLD_DAYS] (same as your earlier backtest).

    Returns: (net_ret, mae, mfe) measured from entry_fill.
    MAE/MFE computed from extremes between entry day..exit day inclusive (on daily bars).
    Conservative: if SL and TP both hit on same day, assume SL first.
    """
    slip = SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0

    entry_i = sig_i + 1
    time_exit_i = sig_i + HOLD_DAYS

    entry_open = float(open_[entry_i])
    entry_fill = entry_open + slip

    # ATR% from signal day (t). Use pct relative to price; apply to entry level
    a_pct = float(atr_pct[sig_i])
    # If ATR% missing/zero, bail
    if not np.isfinite(a_pct) or a_pct <= 0:
        return np.nan, np.nan, np.nan

    sl_level = entry_open * (1.0 - sl_k * a_pct) if sl_k > 0 else -np.inf
    tp_level = entry_open * (1.0 + tp_k * a_pct) if tp_k > 0 else np.inf

    exit_i = time_exit_i
    exit_price_level = float(close[time_exit_i])  # default time exit at close

    # scan from entry day through time_exit day
    for j in range(entry_i, time_exit_i + 1):
        lo = float(low[j])
        hi = float(high[j])

        hit_sl = (sl_k > 0) and (lo <= sl_level)
        hit_tp = (tp_k > 0) and (hi >= tp_level)

        if hit_sl and hit_tp:
            # Conservative tie-break on daily candles
            exit_i = j
            exit_price_level = sl_level
            break
        if hit_sl:
            exit_i = j
            exit_price_level = sl_level
            break
        if hit_tp:
            exit_i = j
            exit_price_level = tp_level
            break

    # Apply slippage on exit: selling gets worse price
    exit_fill = exit_price_level - slip

    # Commission (round trip): subtract from return
    # Convert USD commission to return by dividing by entry_fill
    net = (exit_fill / entry_fill) - 1.0
    if COMMISSION_PER_TRADE_USD != 0.0:
        net -= (2.0 * COMMISSION_PER_TRADE_USD) / entry_fill

    # MAE/MFE until exit day (inclusive), based on fills relative to entry_fill
    window_low = np.min(low[entry_i:exit_i + 1])
    window_high = np.max(high[entry_i:exit_i + 1])

    mae = (float(window_low) / entry_fill) - 1.0
    mfe = (float(window_high) / entry_fill) - 1.0

    return net, mae, mfe


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # sanity required cols
    needed = {"ticker","date","open","high","low","close","ret1d","ret5d","atr14_pct","sma200","sma200_slope20"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in parquet: {missing}")

    # Pre-split by ticker into numpy arrays + signal indices
    ticker_data = {}
    total_signals = 0

    for tkr, g in df.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        sig = build_tp_v2_signal(g)
        sig_idx = np.where(sig.to_numpy(bool))[0]

        if len(sig_idx) == 0:
            continue

        ticker_data[tkr] = dict(
            open=g["open"].to_numpy(float),
            high=g["high"].to_numpy(float),
            low=g["low"].to_numpy(float),
            close=g["close"].to_numpy(float),
            atr_pct=g["atr14_pct"].to_numpy(float),
            sig_idx=sig_idx,
        )
        total_signals += len(sig_idx)

    print(f"TP_v2 signals found (hold={HOLD_DAYS} convention): {total_signals:,}")

    rows = []

    # Sweep configs
    for sl_k in SL_K_LIST:
        for tp_k in TP_K_LIST:
            # Skip "both off" duplicates? keep (0,0) as baseline time-exit.
            ret_list = []
            mae_list = []
            mfe_list = []

            for tkr, d in ticker_data.items():
                for sig_i in d["sig_idx"]:
                    net, mae, mfe = simulate_one(
                        d["open"], d["high"], d["low"], d["close"], d["atr_pct"],
                        sig_i=int(sig_i),
                        sl_k=float(sl_k),
                        tp_k=float(tp_k),
                    )
                    if np.isfinite(net):
                        ret_list.append(net)
                        mae_list.append(mae)
                        mfe_list.append(mfe)

            ret_arr = np.array(ret_list, dtype=float)
            mae_arr = np.array(mae_list, dtype=float)
            mfe_arr = np.array(mfe_list, dtype=float)

            stats = summarize(ret_arr, mae_arr, mfe_arr)
            rows.append({
                "hold_days": HOLD_DAYS,
                "sl_k_atr": sl_k,
                "tp_k_atr": tp_k,
                **stats
            })

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}\n")

    # Print top configs by mean_ret with an N floor
    filt = res[res["n"] >= MIN_N].copy()
    if filt.empty:
        print(f"No configs met MIN_N={MIN_N}. Lower MIN_N.")
        return

    top = filt.sort_values("mean_ret", ascending=False).head(12)
    print(f"Top exit configs by mean_ret (n>={MIN_N}):")
    print(top[[
        "sl_k_atr","tp_k_atr","n","win_rate","mean_ret","median_ret","p25","p75","mean_mae","mean_mfe"
    ]].to_string(index=False))

    # Also show top configs by least adverse MAE (risk control)
    top_mae = filt.sort_values("mean_mae", ascending=False).head(12)  # less negative is better
    print(f"\nTop exit configs by BEST mean_mae (n>={MIN_N}, less negative is better):")
    print(top_mae[[
        "sl_k_atr","tp_k_atr","n","win_rate","mean_ret","median_ret","mean_mae","mean_mfe"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
