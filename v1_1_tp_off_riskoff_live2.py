import os
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import BDay

PARQUET_PATH = "data_live/sp100_daily_features.parquet"
QQQ_PATH = "data_live/qqq_dd52w.parquet"  # must have date, qqq_close

OUT_DIR = "orders"
STATE_POSITIONS_CSV = "state/positions_live.csv"  # create this file (can be empty with headers)

# --- v1.1 config ---
MAX_TOTAL_SLOTS = 30
SLOTS = {"TP_v2": 18, "MR_1A": 12, "EB_1A": 0}
CAP_W = {"TP_v2": 0.60, "MR_1A": 0.40, "EB_1A": 0.00}

TP_HOLD = 20
TP_RET5D_MAX = -0.04
TP_ATR_MIN = 0.02

MR_HOLD = 10
MR_RET3D_MAX = -0.07
MR_ATR_MIN = 0.02

PAPER_EQUITY_USD = 10000  # paper account size

def next_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    # Handles weekends. We'll upgrade to NYSE holidays later if needed.
    return (pd.Timestamp(d).normalize() + BDay(1)).normalize()

def build_qqq_regime_map(qqq_path: str) -> dict:
    q = pd.read_parquet(qqq_path).copy()
    q["date"] = pd.to_datetime(q["date"])
    if "qqq_close" not in q.columns:
        raise RuntimeError(f"QQQ file missing 'qqq_close'. Columns: {q.columns.tolist()}")
    q = q.sort_values("date").reset_index(drop=True)
    q["qqq_sma200"] = q["qqq_close"].rolling(200, min_periods=200).mean()
    q["risk_on"] = q["qqq_close"] > q["qqq_sma200"]
    return {pd.Timestamp(d).normalize(): bool(v) for d, v in zip(q["date"], q["risk_on"])}

def tp_signal(row) -> bool:
    if not (np.isfinite(row.sma200) and np.isfinite(row.sma200_slope20)):
        return False
    if row.sma200_slope20 <= 0:
        return False
    if row.close <= row.sma200:
        return False
    if not np.isfinite(row.ret5d) or row.ret5d > TP_RET5D_MAX:
        return False
    if not np.isfinite(row.ret1d) or row.ret1d > 0:
        return False
    if not np.isfinite(row.atr14_pct) or row.atr14_pct < TP_ATR_MIN:
        return False
    return True

def mr_signal(row) -> bool:
    if not np.isfinite(row.ret3d) or row.ret3d > MR_RET3D_MAX:
        return False
    if not np.isfinite(row.sma10) or row.close >= row.sma10:
        return False
    if not np.isfinite(row.atr14_pct) or row.atr14_pct < MR_ATR_MIN:
        return False
    return True

def tp_rank(row) -> float:
    return float(-row.ret5d)

def mr_rank(row) -> float:
    return float(-row.ret3d)

def load_positions_state(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(columns=["ticker","strategy","entry_date","exit_date","shares"]).to_csv(path, index=False)
    pos = pd.read_csv(path)
    if len(pos) == 0:
        return pos
    pos["entry_date"] = pd.to_datetime(pos["entry_date"])
    pos["exit_date"] = pd.to_datetime(pos["exit_date"])
    return pos

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date","ticker"]).reset_index(drop=True)

    # Use the most recent date in your dataset as the SIGNAL date (e.g., Friday on a weekend)
    signal_date = pd.Timestamp(df["date"].max()).normalize()

    # Next trading day is the intended ENTRY date (e.g., Monday if signal_date is Friday)
    entry_date = next_trading_day(signal_date)

    qqq_risk_on = build_qqq_regime_map(QQQ_PATH)
    risk_on = qqq_risk_on.get(signal_date.normalize(), True)

    # Load current positions (paper state)
    pos = load_positions_state(STATE_POSITIONS_CSV)

    # Determine active slots and held tickers
    active_tp = int((pos["strategy"] == "TP_v2").sum()) if len(pos) else 0
    active_mr = int((pos["strategy"] == "MR_1A").sum()) if len(pos) else 0
    held = set(pos["ticker"].astype(str).tolist()) if len(pos) else set()

    avail_tp = max(0, SLOTS["TP_v2"] - active_tp)
    avail_mr = max(0, SLOTS["MR_1A"] - active_mr)

    # Slice SIGNAL day's rows (do not require entry_date to exist in dataset)
    today_df = df[df["date"] == signal_date].copy()

    # Remove already-held tickers (one position per ticker)
    today_df = today_df[~today_df["ticker"].astype(str).isin(held)].copy()


    # Candidates
    tp_cands = []
    mr_cands = []
    for _, row in today_df.iterrows():
        if tp_signal(row):
            tp_cands.append((row["ticker"], tp_rank(row), float(row["atr14_pct"]), float(row["close"])))
        if mr_signal(row):
            mr_cands.append((row["ticker"], mr_rank(row), float(row["atr14_pct"]), float(row["close"])))

    tp_cands.sort(key=lambda x: x[1], reverse=True)
    mr_cands.sort(key=lambda x: x[1], reverse=True)

    tp_pick = tp_cands[:avail_tp]
    mr_pick = mr_cands[:avail_mr]

    # TP gate: if risk-off, disable TP entries
    tp_enabled = bool(risk_on)
    if not tp_enabled:
        tp_pick = []

    # Simple per-trade $ sizing (paper): weight by strategy budget / slots
    # (We use close as estimate; you will execute with a limit order next open.)
    budget_tp = PAPER_EQUITY_USD * CAP_W["TP_v2"]
    budget_mr = PAPER_EQUITY_USD * CAP_W["MR_1A"]

    per_tp = (budget_tp / SLOTS["TP_v2"]) if SLOTS["TP_v2"] > 0 else 0.0
    per_mr = (budget_mr / SLOTS["MR_1A"]) if SLOTS["MR_1A"] > 0 else 0.0

    orders = []
    for (tkr, score, atr, close) in tp_pick:
        shares = int(np.floor(per_tp / close)) if close > 0 else 0
        if shares <= 0:
            continue
        orders.append({
            "date_generated": str(signal_date.date()),
            "intended_entry_date": str(entry_date.date()),
            "ticker": tkr,
            "strategy": "TP_v2",
            "shares": shares,
            "hold_days": TP_HOLD,
            "planned_exit_date": "",  # optional: fill from trading calendar later
            "rank_score": score,
            "atr14_pct": atr,
            "qqq_risk_on": risk_on,
            "tp_enabled": tp_enabled,
            "ref_close": close,
            "entry_limit_rule": "limit = ref_close * 1.01 (active 9:30-9:35 ET)"
        })

    for (tkr, score, atr, close) in mr_pick:
        shares = int(np.floor(per_mr / close)) if close > 0 else 0
        if shares <= 0:
            continue
        orders.append({
            "date_generated": str(signal_date.date()),
            "intended_entry_date": str(entry_date.date()),
            "ticker": tkr,
            "strategy": "MR_1A",
            "shares": shares,
            "hold_days": MR_HOLD,
            "planned_exit_date": "",
            "rank_score": score,
            "atr14_pct": atr,
            "qqq_risk_on": risk_on,
            "tp_enabled": tp_enabled,
            "ref_close": close,
            "entry_limit_rule": "limit = ref_close * 1.01 (active 9:30-9:35 ET)"
        })

    orders_df = pd.DataFrame(orders).sort_values(["strategy","rank_score"], ascending=[True, False])
    out_path = os.path.join(OUT_DIR, f"orders_{signal_date.date()}.csv")
    orders_df.to_csv(out_path, index=False)

    print("LIVE v1.1 | TP_off_riskoff | 60/40 | 30 slots")
    print(f"Today: {signal_date.date()}  Tomorrow: {entry_date.date()}")
    print(f"QQQ regime: {'RISK-ON' if risk_on else 'RISK-OFF'} (TP enabled={tp_enabled})")
    print(f"Active: TP {active_tp}/{SLOTS['TP_v2']} | MR {active_mr}/{SLOTS['MR_1A']} | Total {len(pos)}/{MAX_TOTAL_SLOTS}")
    print(f"New orders for tomorrow open: TP {len(tp_pick)} | MR {len(mr_pick)}")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
