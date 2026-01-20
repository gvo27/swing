import numpy as np
import pandas as pd
import itertools

# =========================
# Paths
# =========================
STOCK_PATH = "data/sp100_daily_features.parquet"
QQQ_PATH   = "data/qqq_dd52w.parquet"

# =========================
# Load data
# =========================
df = pd.read_parquet(STOCK_PATH)
df["date"] = pd.to_datetime(df["date"]).dt.normalize()
df = df.sort_values(["ticker","date"]).reset_index(drop=True)

print(f"Loaded stock rows: {len(df)} tickers: {df['ticker'].nunique()} dates: {df['date'].nunique()}")

# =========================
# Build QQQ features
# =========================
qqq = pd.read_parquet(QQQ_PATH).copy()
qqq["date"] = pd.to_datetime(qqq["date"]).dt.normalize()
qqq = qqq.sort_values("date").reset_index(drop=True)

qqq["qqq_sma200"]  = qqq["qqq_close"].rolling(200, min_periods=200).mean()
qqq["risk_on"]     = qqq["qqq_close"] > qqq["qqq_sma200"]
qqq["qqq_ret20d"]  = qqq["qqq_close"].pct_change(20)

if "qqq_dd52w" not in qqq.columns:
    roll_high = qqq["qqq_close"].rolling(252, min_periods=252).max()
    qqq["qqq_dd52w"] = (qqq["qqq_close"] / roll_high) - 1.0

qqq = qqq[["date","risk_on","qqq_ret20d","qqq_dd52w"]]

df = df.merge(qqq, on="date", how="left")

# =========================
# TP signal
# =========================
def tp_signal(row, ret5d_max, atr_min, min_above_sma200):
    if not np.isfinite(row.ret5d) or row.ret5d > ret5d_max:
        return False
    if not np.isfinite(row.ret1d) or row.ret1d > 0:
        return False
    if not np.isfinite(row.atr14_pct) or row.atr14_pct < atr_min:
        return False
    if not np.isfinite(row.sma200) or row.close <= row.sma200 * (1 + min_above_sma200):
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
        return bool(row.risk_on) and np.isfinite(row.qqq_ret20d) and row.qqq_ret20d >= threshold

    if gate_mode == "dd52w_above":
        return np.isfinite(row.qqq_dd52w) and row.qqq_dd52w >= threshold

    raise ValueError(gate_mode)

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
    "risk_on_and_ret20d": [-0.02, 0.00, 0.02],
    "dd52w_above": [-0.20, -0.15, -0.10]
}

# =========================
# Precompute forward returns
# =========================
df["fwd_ret"] = (
    df.groupby("ticker")["close"]
      .shift(-20) / df["open"].shift(-1) - 1.0
)

df["year"] = df["date"].dt.year

# =========================
# Sweep
# =========================
results = []
configs = list(itertools.product(HOLDS, RET5D_MAX, ATR_MIN, MIN_ABOVE))
print(f"Starting sweep... base TP configs: {len(configs)}")

for hold, ret5d_max, atr_min, min_above in configs:
    base_mask = df.apply(
        lambda r: tp_signal(r, ret5d_max, atr_min, min_above),
        axis=1
    )

    for gate_mode, thresholds in GATES.items():
        for thr in thresholds:
            mask = base_mask & df.apply(
                lambda r: gate_pass(r, gate_mode, thr),
                axis=1
            )

            picks = df[mask & df["fwd_ret"].notna()]

            if len(picks) < 200:
                continue

            all_mean = picks["fwd_ret"].mean()
            all_win  = (picks["fwd_ret"] > 0).mean()

            y2022 = picks[picks["year"] == 2022]
            if len(y2022) < 30:
                continue

            results.append({
                "hold": hold,
                "gate_mode": gate_mode,
                "threshold": thr,
                "ret5d_max": ret5d_max,
                "atr_min": atr_min,
                "min_above_sma200": min_above,
                "n_all": len(picks),
                "mean_all": all_mean,
                "win_all": all_win,
                "n_2022": len(y2022),
                "mean_2022": y2022["fwd_ret"].mean(),
                "win_2022": (y2022["fwd_ret"] > 0).mean(),
            })

# =========================
# Results
# =========================
res = pd.DataFrame(results)
res = res.sort_values("mean_2022", ascending=False)

print("\n=== Top TP_v3 configs (ranked by 2022 mean_ret) ===")
print(res.head(20).to_string(index=False))

print("\n=== Best config per gate_mode ===")
for gm in res["gate_mode"].unique():
    best = res[res["gate_mode"] == gm].iloc[0]
    print("\n", gm)
    print(best.to_string())
