import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"

# -----------------------------
# Portfolio config
# -----------------------------
MAX_TOTAL_SLOTS = 30
SLOTS = {"TP_v2": 18, "MR_1A": 12, "EB_1A": 0}

CAP_W = {"TP_v2": 0.60, "MR_1A": 0.40, "EB_1A": 0.10}

START_CAPITAL = 1.0
ALLOW_MULTIPLE_POS_PER_TICKER = False  # recommended

END_DATE = pd.Timestamp("2025-12-22")

QQQ_PATH = "data/qqq_dd52w.parquet"
USE_QQQ_GATE_FOR_TP = True

# -----------------------------
# Strategy params (locked)
# -----------------------------
TP_HOLD = 20
TP_RET5D_MAX = -0.04
TP_ATR_MIN = 0.02

MR_HOLD = 10
MR_RET3D_MAX = -0.07
MR_ATR_MIN = 0.02

EB_HOLD = 10
EB_GAP_MIN = 0.04
EB_ATR_MIN = 0.02

# -----------------------------
# Execution assumptions
# -----------------------------
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0
COMMISSION_PER_TRADE_USD = 0.0  # keep 0 for now

def entry_fill(open_px: float) -> float:
    return float(open_px) + (SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0)

def exit_fill(close_px: float) -> float:
    return float(close_px) - (SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0)

# -----------------------------
# Features
# -----------------------------
def ensure_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    if "ret3d" not in g.columns:
        g["ret3d"] = g["close"] / g["close"].shift(3) - 1.0
    if "sma10" not in g.columns:
        g["sma10"] = g["close"].rolling(10, min_periods=10).mean()
    g["prev_close"] = g["close"].shift(1)
    g["gap"] = (g["open"] / g["prev_close"]) - 1.0
    g["day_ret"] = (g["close"] / g["open"]) - 1.0
    return g

def build_qqq_gate(end_date: pd.Timestamp) -> dict:
    q = pd.read_parquet(QQQ_PATH).copy()
    q["date"] = pd.to_datetime(q["date"])
    q = q[q["date"] <= end_date].sort_values("date").reset_index(drop=True)

    # accept either qqq_close or close
    if "qqq_close" in q.columns:
        px = "qqq_close"
    elif "close" in q.columns:
        px = "close"
    else:
        raise RuntimeError(f"QQQ file missing close column. Columns: {q.columns.tolist()}")

    q["sma200"] = q[px].rolling(200, min_periods=200).mean()
    q["risk_on"] = q[px] > q["sma200"]

    # map: date -> bool (risk_on)
    return {pd.Timestamp(d).normalize(): bool(v) for d, v in zip(q["date"], q["risk_on"])}


# -----------------------------
# Signals + ranks (EOD signal, enter next open)
# -----------------------------
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

def tp_rank(row) -> float:
    return float(-row.ret5d)

def mr_signal(row, risk_on: bool) -> bool:
    # stricter oversold requirement in risk-off (match script #2)
    ret3d_cut = MR_RET3D_MAX if risk_on else -0.09
    if not np.isfinite(row.ret3d) or row.ret3d > ret3d_cut:
        return False
    if not np.isfinite(row.sma10) or row.close >= row.sma10:
        return False
    if not np.isfinite(row.atr14_pct) or row.atr14_pct < MR_ATR_MIN:
        return False
    return True

def mr_rank(row) -> float:
    return float(-row.ret3d)

def eb_signal(row) -> bool:
    if not np.isfinite(row.gap) or row.gap < EB_GAP_MIN:
        return False
    if not np.isfinite(row.day_ret) or row.day_ret < 0:
        return False
    if not np.isfinite(row.atr14_pct) or row.atr14_pct < EB_ATR_MIN:
        return False
    if not np.isfinite(row.sma200) or row.close <= row.sma200:
        return False
    return True

def eb_rank(row) -> float:
    return float(row.gap)

# -----------------------------
# Position object
# -----------------------------
class Position:
    __slots__ = ("strategy","ticker","entry_date","exit_date","shares")
    def __init__(self, strategy, ticker, entry_date, exit_date, shares):
        self.strategy = strategy
        self.ticker = ticker
        self.entry_date = pd.Timestamp(entry_date)
        self.exit_date = pd.Timestamp(exit_date)
        self.shares = float(shares)

# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= END_DATE]
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)
    df = df.groupby("ticker", sort=False, group_keys=False).apply(ensure_features)

    # minimal sanity
    required = {"date","ticker","open","close","high","low","atr14_pct","sma200","sma200_slope20","ret1d","ret5d","ret3d","sma10","gap","day_ret"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    date_to_i = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    qqq_risk_on = {}
    if USE_QQQ_GATE_FOR_TP:
        qqq_risk_on = build_qqq_gate(END_DATE)

    frames_by_date = {pd.Timestamp(d): g.set_index("ticker", drop=False) for d, g in df.groupby("date", sort=True)}

    cash = START_CAPITAL
    positions = []  # list[Position]

    equity_curve = []
    daily_returns = []

    def current_equity(today_frame):
        val = cash
        for p in positions:
            if p.ticker in today_frame.index:
                val += p.shares * float(today_frame.loc[p.ticker, "close"])
        return val

    for di, d in enumerate(dates):
        day = pd.Timestamp(d)
        today = frames_by_date.get(day)
        if today is None or today.empty:
            continue

        eq_prev = equity_curve[-1]["equity"] if equity_curve else START_CAPITAL

        # 1) process exits at today's close
        still = []
        for p in positions:
            if p.exit_date == day:
                if p.ticker in today.index:
                    px = exit_fill(float(today.loc[p.ticker, "close"]))
                    cash_add = p.shares * px
                    cash_add -= COMMISSION_PER_TRADE_USD
                    cash += cash_add
            else:
                still.append(p)
        positions = still

        # 2) mark equity at close
        eq = current_equity(today)
        r = (eq / eq_prev - 1.0) if eq_prev > 0 else 0.0

        equity_curve.append({"date": day, "equity": eq})
        daily_returns.append({"date": day, "ret": r})

        # 3) generate entries from today's signals for NEXT day open
        if di >= len(dates) - 1:
            continue
        next_day = pd.Timestamp(dates[di + 1])
        tomorrow = frames_by_date.get(next_day)
        if tomorrow is None or tomorrow.empty:
            continue

        # slot availability (per strategy)
        active_tp = sum(1 for p in positions if p.strategy == "TP_v2")
        active_mr = sum(1 for p in positions if p.strategy == "MR_1A")
        active_eb = sum(1 for p in positions if p.strategy == "EB_1A")

        avail = {
            "TP_v2": max(0, SLOTS["TP_v2"] - active_tp),
            "MR_1A": max(0, SLOTS["MR_1A"] - active_mr),
            "EB_1A": max(0, SLOTS["EB_1A"] - active_eb),
        }

        held_tickers = {p.ticker for p in positions} if not ALLOW_MULTIPLE_POS_PER_TICKER else set()

        # candidates must exist tomorrow (need open)
        common = today.index.intersection(tomorrow.index)

        tp_cands, mr_cands, eb_cands = [], [], []

        # QQQ regime (match script #2 behavior)
        risk_on = True
        if USE_QQQ_GATE_FOR_TP:
            risk_on = bool(qqq_risk_on.get(day.normalize(), True))

        # TP gate mode is "off" in risk-off -> TP only allowed when risk_on
        tp_allowed_today = risk_on

        for tkr in common:
            if (not ALLOW_MULTIPLE_POS_PER_TICKER) and (tkr in held_tickers):
                continue
            row = today.loc[tkr]
            atr = float(row.atr14_pct) if np.isfinite(row.atr14_pct) else np.nan
            if not np.isfinite(atr) or atr <= 0:
                continue

            if tp_allowed_today and tp_signal(row):
                tp_cands.append((tkr, tp_rank(row), atr))
            if mr_signal(row, risk_on):
                mr_cands.append((tkr, mr_rank(row), atr))
            if eb_signal(row):
                eb_cands.append((tkr, eb_rank(row), atr))

        tp_cands.sort(key=lambda x: x[1], reverse=True)
        mr_cands.sort(key=lambda x: x[1], reverse=True)
        eb_cands.sort(key=lambda x: x[1], reverse=True)

        tp_pick = tp_cands[:avail["TP_v2"]]
        mr_pick = mr_cands[:avail["MR_1A"]]
        eb_pick = eb_cands[:avail["EB_1A"]]

        # allocate budgets based on CURRENT equity (not cash), but spend from cash
        eq_now = eq
        budgets = {}
        for strat, picks in [("TP_v2", tp_pick), ("MR_1A", mr_pick), ("EB_1A", eb_pick)]:
            budgets[strat] = eq_now * CAP_W[strat] if len(picks) > 0 else 0.0

        # ensure we don't spend more cash than we have
        total_budget = sum(budgets.values())
        if total_budget > cash and total_budget > 0:
            scale = cash / total_budget
            for k in budgets:
                budgets[k] *= scale

        # helper to place orders
        def enter_positions(strat, picks, hold_days):
            nonlocal cash, positions
            if len(picks) == 0:
                return
            budget = budgets[strat]
            if budget <= 0:
                return

            inv_atr = np.array([1.0 / max(1e-6, p[2]) for p in picks], dtype=float)
            w = inv_atr / inv_atr.sum()

            for (tkr, _score, _atr), wi in zip(picks, w):
                px_open = float(tomorrow.loc[tkr, "open"])
                fill = entry_fill(px_open)
                if not np.isfinite(fill) or fill <= 0:
                    continue

                dollars = budget * float(wi)
                cost = dollars + COMMISSION_PER_TRADE_USD
                if cost > cash:
                    continue  # insufficient cash
                shares = dollars / fill

                exit_idx = date_to_i[next_day] + hold_days
                if exit_idx >= len(dates):
                    continue
                exit_date = pd.Timestamp(dates[exit_idx])

                cash -= cost
                positions.append(Position(strat, tkr, next_day, exit_date, shares))

        enter_positions("TP_v2", tp_pick, TP_HOLD)
        enter_positions("MR_1A", mr_pick, MR_HOLD)
        enter_positions("EB_1A", eb_pick, EB_HOLD)

    eq = pd.DataFrame(equity_curve).set_index("date")
    rets = pd.DataFrame(daily_returns).set_index("date")

    total_ret = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    roll_max = eq["equity"].cummax()
    dd = eq["equity"] / roll_max - 1.0
    max_dd = float(dd.min())

    eq_y = eq["equity"].resample("Y").last()
    yret = (eq_y / eq_y.shift(1) - 1.0).dropna()
    yret.index = yret.index.year

    print("\n=== 30-slot CASH Portfolio Backtest (TP/MR/EB) ===")
    print(f"Start: {eq.index[0].date()}   End: {eq.index[-1].date()}   Years: {years:.2f}")
    print(f"Total return: {total_ret:.2%}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Avg daily ret: {rets['ret'].mean():.5f}   Daily vol: {rets['ret'].std():.5f}")

    print("\nYearly returns:")
    print(yret.to_string())

    # Save outputs
    eq_path = os.path.join(OUT_DIR, "portfolio30_cash_equity_curve.csv")
    dd_path = os.path.join(OUT_DIR, "portfolio30_cash_drawdown.csv")
    rets_path = os.path.join(OUT_DIR, "portfolio30_cash_daily_returns.csv")

    eq.to_csv(eq_path)
    dd.to_frame("drawdown").to_csv(dd_path)
    rets.to_csv(rets_path)

    print("\nSaved:")
    print(f"  {eq_path}")
    print(f"  {dd_path}")
    print(f"  {rets_path}")

if __name__ == "__main__":
    main()
