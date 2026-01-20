# tp_v3_gate_sweep_v2.py
# Purpose: Fast TP_v3 research sweep with correct next-open -> (hold+1) close execution,
#          plus market regime gates (none / risk_on / risk_on_and_ret20d / dd52w_above).
#
# Run:
#   python tp_v3_gate_sweep_v2.py
#
# Output:
#   Prints top configs ranked by 2022 mean return + best config per gate mode.

import numpy as np
import pandas as pd
import itertools

# =========================
# Paths
# =========================
STOCK_PATH = "data/sp100_daily_features.parquet"
QQQ_PATH   = "data/qqq_dd52w.parquet"

# =========================
# Config
# =========================
START_DATE = "2015-10-16"
END_DATE   = "2025-12-22"

# Discovery thresholds (looser so gated configs show up)
MIN_N_ALL  = 120
MIN_N_2022 = 20

# =========================
# Load stock data
# =========================
df = pd.read_parquet(STOCK_PATH)
df["date"] = pd.to_datetime(df["date"]).dt.normalize()
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

# Optional date clip for comparability with your frozen window
df = df[(df["date"] >= pd.Timestamp(START_DATE)) & (df["date"] <= pd.Timestamp(END_DATE))].copy()

print(f"Loaded stock rows: {len(df)} | tickers: {df['ticker'].nunique()} | dates: {df['date'].nunique()}")

# =========================
# Build QQQ features (risk_on, ret20d, dd52w)
# =========================
qqq = pd.read_parquet(QQQ_PATH).copy()
qqq["date"] = pd.to_datetime(qqq["date"]).dt.normalize()
qqq = qqq.sort_values("date").reset_index(drop=True)

if "qqq_close" not in qqq.columns:
    raise RuntimeError(f"QQQ file missing 'qqq_close'. Columns: {qqq.columns.tolist()}")

qqq["qqq_sma200"] = qqq["qqq_close"].rolling(200, min_periods=200).mean()
qqq["risk_on"] = qqq["qqq_close"] > qqq["qqq_sma200"]
qqq["qqq_ret20d"] = qqq["qqq_close"].pct_change(20)

# Accept precomputed drawdown if present, else compute
cols = set(qqq.columns)
if "qqq_dd52w" in cols:
    qqq["qqq_dd52w"] = qqq["qqq_dd52w"].astype(float)
elif "dd_from_52w_high" in cols:
    qqq["qqq_dd52w"] = qqq["dd_from_52w_high"].astype(float)
else:
    roll_high = qqq["qqq_close"].rolling(252, min_periods=252).max()
    qqq["qqq_dd52w"] = (qqq["qqq_close"] / roll_high) - 1.0

qqq = qqq[["date", "risk_on", "qqq_ret20d", "qqq_dd52w"]]

# Merge QQQ features onto stock rows by date
df = df.merge(qqq, on="date", how="left")

# =========================
# Required columns check
# =========================
required = {
    "ticker", "date", "open", "close",
    "ret1d", "ret5d",
    "atr14_pct", "sma200", "sma200_slope20",
    "risk_on", "qqq_ret20d", "qqq_dd52w"
}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns in stock dataframe: {missing}")

df["year"] = df["date"].dt.year

# =========================
# TP signal (close of day i -> enter next open)
# =========================
def tp_signal(row, ret5d_max, atr_min, min_above_sma200):
    # trend must be up
    if not np.isfinite(row.sma200_slope20) or row.sma200_slope20 <= 0:
        return False

    # above sma200 by at least X
    if not np.isfinite(row.sma200) or row.close <= row.sma200 * (1.0 + min_above_sma200):
        return False

    # pullback conditions
    if not np.isfinite(row.ret5d) or row.ret5d > ret5d_max:
        return False
    if not np.isfinite(row.ret1d) or row.ret1d > 0:
        return False

    # volatility floor
    if not np.isfinite(row.atr14_pct) or row.atr14_pct < atr_min:
        return False

    return True

# =========================
# Gate logic
# =========================
def gate_pass(row, gate_mode, threshold):
    if gate_mode == "none":
        return True

    if gate_mode == "risk_on":
        return bool(row.risk_on)

    if gate_mode == "risk_on_and_ret20d":
        return bool(row.risk_on) and np.isfinite(row.qqq_ret20d) and (row.qqq_ret20d >= float(threshold))

    if gate_mode == "dd52w_above":
        return np.isfinite(row.qqq_dd52w) and (row.qqq_dd52w >= float(threshold))

    raise ValueError(f"Unknown gate_mode: {gate_mode}")

# =========================
# Parameter grid
# =========================
HOLDS = [20]

RET5D_MAX = [-0.04, -0.05]
ATR_MIN   = [0.02, 0.025, 0.03]
MIN_ABOVE = [0.00, 0.02]

GATES = {
    "none": [None],
    "risk_on": [None],
    "risk_on_and_ret20d": [-0.02, 0.00, 0.02, 0.05],
    "dd52w_above": [-0.15, -0.10, -0.05, 0.00],
}

# =========================
# Sweep (vectorized where it matters)
# =========================
results = []

base_configs = list(itertools.product(HOLDS, RET5D_MAX, ATR_MIN, MIN_ABOVE))
print(f"Starting sweep... base TP configs: {len(base_configs)}")

# Precompute next-day open per ticker once
open_next = df.groupby("ticker")["open"].shift(-1)

for hold, ret5d_max, atr_min, min_above in base_configs:
    # Correct forward return alignment:
    # signal at close(t) -> enter open(t+1) -> exit close(t+1+hold)
    close_exit = df.groupby("ticker")["close"].shift(-(hold + 1))
    fwd_ret = (close_exit / open_next) - 1.0

    # Base TP condition (row-wise via apply; ok because config count is small)
    base_mask = df.apply(lambda r: tp_signal(r, ret5d_max, atr_min, min_above), axis=1)

    for gate_mode, thresholds in GATES.items():
        for thr in thresholds:
            gate_mask = df.apply(lambda r: gate_pass(r, gate_mode, thr), axis=1)
            mask = base_mask & gate_mask & fwd_ret.notna()

            picks = df.loc[mask, ["date", "year"]].copy()
            if picks.empty:
                continue

            # Attach returns
            picks["fwd_ret"] = fwd_ret.loc[mask].astype(float).values

            n_all = int(len(picks))
            if n_all < MIN_N_ALL:
                continue

            all_mean = float(picks["fwd_ret"].mean())
            all_win  = float((picks["fwd_ret"] > 0).mean())

            y2022 = picks[picks["year"] == 2022]
            n_2022 = int(len(y2022))
            if n_2022 < MIN_N_2022:
                continue

            mean_2022 = float(y2022["fwd_ret"].mean())
            win_2022  = float((y2022["fwd_ret"] > 0).mean())

            results.append({
                "hold": hold,
                "gate_mode": gate_mode,
                "threshold": (np.nan if thr is None else float(thr)),
                "ret5d_max": float(ret5d_max),
                "atr_min": float(atr_min),
                "min_above_sma200": float(min_above),
                "n_all": n_all,
                "mean_all": all_mean,
                "win_all": all_win,
                "n_2022": n_2022,
                "mean_2022": mean_2022,
                "win_2022": win_2022,
            })

# =========================
# Results
# =========================
if len(results) == 0:
    print("\nNo configs met thresholds. Try lowering MIN_N_ALL / MIN_N_2022.")
    raise SystemExit(0)

res = pd.DataFrame(results).sort_values("mean_2022", ascending=False).reset_index(drop=True)

print("\n=== Top TP_v3 configs (ranked by 2022 mean_ret) ===")
print(res.head(25).to_string(index=False))

print("\n=== Best config per gate_mode (by mean_2022) ===")
for gm in ["none", "risk_on", "risk_on_and_ret20d", "dd52w_above"]:
    sub = res[res["gate_mode"] == gm]
    if len(sub) == 0:
        print(f"\n {gm}\n  (no configs met thresholds)")
    else:
        print(f"\n {gm}")
        print(sub.head(5).to_string(index=False))
