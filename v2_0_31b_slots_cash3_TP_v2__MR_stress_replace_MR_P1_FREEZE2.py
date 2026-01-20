# v2_0_31b_slots_cash3_TP_v2__MR_stress_replace_MR_P1_FREEZE.py
import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"

# -----------------------------
# Portfolio config (same as v1.1)
# -----------------------------
MAX_TOTAL_SLOTS = 30
SLOTS = {"TP_v2": 18, "MR_1A": 12, "MR_P1": 2, "EB_1A": 0}
CAP_W = {"TP_v2": 0.60, "MR_1A": 0.35, "MR_P1": 0.05, "EB_1A": 0.00}

START_CAPITAL = 1.0
ALLOW_MULTIPLE_POS_PER_TICKER = False  # recommended

QQQ_PATH = "data/qqq_dd52w.parquet"  # must contain columns: date, qqq_close (dd optional)

# TP regime gating (keep frozen)
TP_GATE_MODE = "off"      # "off" or "reduce"
TP_RISKOFF_SCALE = 0.25   # used only if TP_GATE_MODE == "reduce"

START_DATE = "2015-10-16"
END_DATE   = "2025-12-22"

# -----------------------------
# Strategy params (TP + MR_1A locked)
# -----------------------------
TP_HOLD = 20
TP_RET5D_MAX = -0.04
TP_ATR_MIN = 0.02

MR_HOLD = 10
MR_RET3D_MAX = -0.07
MR_ATR_MIN = 0.02

MRP_HOLD = 5

EB_HOLD = 10
EB_GAP_MIN = 0.04
EB_ATR_MIN = 0.02

# -----------------------------
# New: Stress replacement MR_P1 (locked)
# -----------------------------
USE_MR_P1_ON_STRESS = True
MR_P1_HOLD = 5
MR_P1_RET1D_MAX = -0.05
MR_P1_RET5D_MAX = -0.10
MR_P1_ATR_MIN = 0.03
MR_P1_ALLOW_BELOW_SMA200 = True
MR_STRESS_MODE = "dd52w_leq"     # "dd52w_leq" (for now)
MR_STRESS_THR  = -0.15           # QQQ dd52w <= -0.15

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

def mr_signal_1a(row, risk_on: bool) -> bool:
    # stricter oversold requirement in risk-off
    ret3d_cut = MR_RET3D_MAX if risk_on else -0.09
    if not np.isfinite(row.ret3d) or row.ret3d > ret3d_cut:
        return False
    if not np.isfinite(row.sma10) or row.close >= row.sma10:
        return False
    if not np.isfinite(row.atr14_pct) or row.atr14_pct < MR_ATR_MIN:
        return False
    return True

def mr_rank_1a(row) -> float:
    return float(-row.ret3d)

def mr_signal_p1(row) -> bool:
    # panic MR (stress-only replacement)
    if not np.isfinite(row.ret1d) or row.ret1d > MR_P1_RET1D_MAX:
        return False
    if not np.isfinite(row.ret5d) or row.ret5d > MR_P1_RET5D_MAX:
        return False
    if not np.isfinite(row.atr14_pct) or row.atr14_pct < MR_P1_ATR_MIN:
        return False
    if not MR_P1_ALLOW_BELOW_SMA200:
        if not np.isfinite(row.sma200) or row.close <= row.sma200:
            return False
    return True

def mr_rank_p1(row) -> float:
    # rank deeper panic higher
    return float(-(row.ret1d + row.ret5d))

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
# QQQ regime + stress maps
# -----------------------------
def _load_qqq_df(qqq_path: str) -> pd.DataFrame:
    q = pd.read_parquet(qqq_path).copy()
    # flatten possible multiindex columns
    if isinstance(q.columns, pd.MultiIndex):
        q.columns = ["_".join([str(x) for x in tup if str(x) not in ("", "None")]) for tup in q.columns]
    q.columns = [str(c).strip() for c in q.columns]
    if "date" not in q.columns:
        # sometimes it can be like "('date','')"
        candidates = [c for c in q.columns if "date" in c.lower()]
        if len(candidates) == 1:
            q = q.rename(columns={candidates[0]: "date"})
    q["date"] = pd.to_datetime(q["date"]).dt.normalize()
    return q

def build_qqq_regime_map(qqq_path: str) -> dict:
    q = _load_qqq_df(qqq_path)
    if "qqq_close" not in q.columns:
        raise RuntimeError(f"QQQ file missing 'qqq_close'. Columns: {q.columns.tolist()}")
    q = q.sort_values("date").reset_index(drop=True)
    q["qqq_sma200"] = q["qqq_close"].rolling(200, min_periods=200).mean()
    q["risk_on"] = q["qqq_close"] > q["qqq_sma200"]
    return {pd.Timestamp(d).normalize(): bool(v) for d, v in zip(q["date"], q["risk_on"])}

def build_qqq_dd52w_map(qqq_path: str) -> dict:
    q = _load_qqq_df(qqq_path)
    if "qqq_close" not in q.columns:
        raise RuntimeError(f"QQQ file missing 'qqq_close'. Columns: {q.columns.tolist()}")
    q = q.sort_values("date").reset_index(drop=True)
    if "qqq_dd52w" in q.columns:
        dd = q["qqq_dd52w"].astype(float)
    elif "dd_from_52w_high" in q.columns:
        dd = q["dd_from_52w_high"].astype(float)
    else:
        roll_high = q["qqq_close"].rolling(252, min_periods=252).max()
        dd = (q["qqq_close"] / roll_high) - 1.0
    return {pd.Timestamp(d).normalize(): float(v) for d, v in zip(q["date"], dd)}

def is_stress_day(day: pd.Timestamp, dd52w_map: dict) -> bool:
    if MR_STRESS_MODE == "dd52w_leq":
        v = dd52w_map.get(day.normalize(), np.nan)
        return np.isfinite(v) and (float(v) <= float(MR_STRESS_THR))
    raise RuntimeError(f"Unknown MR_STRESS_MODE={MR_STRESS_MODE}")

# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)
    df = df.groupby("ticker", sort=False, group_keys=False).apply(ensure_features)

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[(df["date"] >= pd.Timestamp(START_DATE)) & (df["date"] <= pd.Timestamp(END_DATE))].copy()

    # minimal sanity
    required = {"date","ticker","open","close","high","low","atr14_pct","sma200","sma200_slope20","ret1d","ret5d","ret3d","sma10","gap","day_ret"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    date_to_i = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    frames_by_date = {pd.Timestamp(d): g.set_index("ticker", drop=False) for d, g in df.groupby("date", sort=True)}

    qqq_risk_on = build_qqq_regime_map(QQQ_PATH)
    qqq_dd52w = build_qqq_dd52w_map(QQQ_PATH)

    # sanity: stress days count
    stress_days = sum(1 for d in dates if is_stress_day(pd.Timestamp(d), qqq_dd52w))
    print(f"\nSanity: stress_days={stress_days} of total_days={len(dates)} (mode={MR_STRESS_MODE}, thr={MR_STRESS_THR})")

    cash = START_CAPITAL
    positions = []  # list[Position]

    equity_curve = []
    daily_returns = []

    # MR sanity counters
    mr_entries_total = 0
    mr_entries_p1 = 0
    mr_entries_1a = 0
    mr_entries_p1_by_year = {}

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

        risk_on = qqq_risk_on.get(day.normalize(), True)
        stress = is_stress_day(day, qqq_dd52w) if USE_MR_P1_ON_STRESS else False

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

        active_mrp = sum(1 for p in positions if p.strategy == "MR_P1")

        avail = {
            "TP_v2": max(0, SLOTS["TP_v2"] - active_tp),
            "MR_1A": max(0, SLOTS["MR_1A"] - active_mr),
            "MR_P1": max(0, SLOTS["MR_P1"] - active_mrp),
            "EB_1A": max(0, SLOTS["EB_1A"] - active_eb),
        }

        held_tickers = {p.ticker for p in positions} if not ALLOW_MULTIPLE_POS_PER_TICKER else set()

        # candidates must exist tomorrow (need open)
        common = today.index.intersection(tomorrow.index)

        tp_cands, mr_cands, mrp_cands, eb_cands = [], [], [], []
        mr_hold_today = MR_HOLD
        mr_using = "MR_1A"

        # conditional replacement
        if USE_MR_P1_ON_STRESS and stress:
            mr_hold_today = MR_P1_HOLD
            mr_using = "MR_P1"

        for tkr in common:
            if (not ALLOW_MULTIPLE_POS_PER_TICKER) and (tkr in held_tickers):
                continue
            row = today.loc[tkr]
            atr = float(row.atr14_pct) if np.isfinite(row.atr14_pct) else np.nan
            if stress:
                if (np.isfinite(row.ret1d) and row.ret1d <= -0.05 and
                    np.isfinite(row.ret5d) and row.ret5d <= -0.10 and
                    np.isfinite(row.atr14_pct) and row.atr14_pct >= 0.03):
                    # rank: more negative ret5d = higher priority
                    mrp_cands.append((tkr, float(-row.ret5d), float(row.atr14_pct)))
            if not np.isfinite(atr) or atr <= 0:
                continue

            if tp_signal(row):
                tp_cands.append((tkr, tp_rank(row), atr))

            # MR candidates: pick one model depending on stress
            if mr_using == "MR_P1":
                if mr_signal_p1(row):
                    mr_cands.append((tkr, mr_rank_p1(row), atr, "MR_P1"))
            else:
                if mr_signal_1a(row, risk_on):
                    mr_cands.append((tkr, mr_rank_1a(row), atr, "MR_1A"))
                # keep your debug line behavior
                if not risk_on and row.ret3d <= -0.07 and row.ret3d > -0.09:
                    print(day.date(), "MR blocked by risk-off gate", row.ret3d)

            if eb_signal(row):
                eb_cands.append((tkr, eb_rank(row), atr))

        tp_cands.sort(key=lambda x: x[1], reverse=True)
        mr_cands.sort(key=lambda x: x[1], reverse=True)
        eb_cands.sort(key=lambda x: x[1], reverse=True)
        mrp_cands.sort(key=lambda x: x[1], reverse=True)
        mrp_pick = mrp_cands[:avail["MR_P1"]]

        tp_pick = tp_cands[:avail["TP_v2"]]
        mr_pick = mr_cands[:avail["MR_1A"]]
        eb_pick = eb_cands[:avail["EB_1A"]]

        # allocate budgets based on CURRENT equity (not cash), but spend from cash
        eq_now = eq

        budgets = {}
        for strat, picks in [("TP_v2", tp_pick), ("MR_1A", mr_pick), ("MR_P1", mrp_pick), ("EB_1A", eb_pick)]:
            if len(picks) == 0:
                budgets[strat] = 0.0
                continue

            base = eq_now * CAP_W[strat]

            if strat == "TP_v2" and (not risk_on):
                if TP_GATE_MODE == "off":
                    base = 0.0
                elif TP_GATE_MODE == "reduce":
                    base = base * float(TP_RISKOFF_SCALE)
                else:
                    raise RuntimeError(f"Unknown TP_GATE_MODE={TP_GATE_MODE}")

            if strat == "MR_P1" and (not stress):
                base = 0.0

            budgets[strat] = base

        # ensure we don't spend more cash than we have
        total_budget = sum(budgets.values())
        if total_budget > cash and total_budget > 0:
            scale = cash / total_budget
            for k in budgets:
                budgets[k] *= scale

        # helper to place orders
        def enter_positions_tp(strat, picks, hold_days):
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
                    continue
                shares = dollars / fill

                exit_idx = date_to_i[next_day] + hold_days
                if exit_idx >= len(dates):
                    continue
                exit_date = pd.Timestamp(dates[exit_idx])

                cash -= cost
                positions.append(Position(strat, tkr, next_day, exit_date, shares))

        def enter_positions_mr(strat, picks, hold_days):
            nonlocal cash, positions, mr_entries_total, mr_entries_p1, mr_entries_1a, mr_entries_p1_by_year
            if len(picks) == 0:
                return
            budget = budgets[strat]
            if budget <= 0:
                return

            inv_atr = np.array([1.0 / max(1e-6, p[2]) for p in picks], dtype=float)
            w = inv_atr / inv_atr.sum()

            for item, wi in zip(picks, w):
                # allow either (tkr, score, atr) or (tkr, score, atr, mr_mode)
                if len(item) == 3:
                    tkr, _score, _atr = item
                    mr_mode = strat  # default label (MR_1A or MR_P1)
                else:
                    tkr, _score, _atr, mr_mode = item
                px_open = float(tomorrow.loc[tkr, "open"])
                fill = entry_fill(px_open)
                if not np.isfinite(fill) or fill <= 0:
                    continue
                dollars = budget * float(wi)
                cost = dollars + COMMISSION_PER_TRADE_USD
                if cost > cash:
                    continue
                shares = dollars / fill

                exit_idx = date_to_i[next_day] + hold_days
                if exit_idx >= len(dates):
                    continue
                exit_date = pd.Timestamp(dates[exit_idx])

                cash -= cost
                positions.append(Position(strat, tkr, next_day, exit_date, shares))

                mr_entries_total += 1
                if mr_mode == "MR_P1":
                    mr_entries_p1 += 1
                    y = int(next_day.year)
                    mr_entries_p1_by_year[y] = mr_entries_p1_by_year.get(y, 0) + 1
                else:
                    mr_entries_1a += 1

        enter_positions_tp("TP_v2", tp_pick, TP_HOLD)
        enter_positions_mr("MR_1A", mr_pick, mr_hold_today)
        enter_positions_tp("EB_1A", eb_pick, EB_HOLD)
        enter_positions_mr("MR_P1", mrp_pick, MRP_HOLD)

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

    print("\n=== 30-slot CASH Portfolio Backtest (TP + MR stress replacement) ===")
    print(f"Start: {eq.index[0].date()}   End: {eq.index[-1].date()}   Years: {years:.2f}")
    print(f"Total return: {total_ret:.2%}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Avg daily ret: {rets['ret'].mean():.5f}   Daily vol: {rets['ret'].std():.5f}")

    print("\nYearly returns:")
    print(yret.to_string())

    print("\nMR entries sanity:")
    print(f"  total: {mr_entries_total} | MR_1A: {mr_entries_1a} | MR_P1: {mr_entries_p1}")
    print(f"  MR_P1 by year: {mr_entries_p1_by_year}")

    # Save outputs
    eq_path = os.path.join(OUT_DIR, "v2_portfolio30_cash_equity_curve.csv")
    dd_path = os.path.join(OUT_DIR, "v2_portfolio30_cash_drawdown.csv")
    rets_path = os.path.join(OUT_DIR, "v2_portfolio30_cash_daily_returns.csv")

    eq.to_csv(eq_path)
    dd.to_frame("drawdown").to_csv(dd_path)
    rets.to_csv(rets_path)

    print("\nSaved:")
    print(f"  {eq_path}")
    print(f"  {dd_path}")
    print(f"  {rets_path}")

if __name__ == "__main__":
    main()
