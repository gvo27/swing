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

CAP_W = {"TP_v2": 0.60, "MR_1A": 0.40, "EB_1A": 0.00}

START_CAPITAL = 1.0
ALLOW_MULTIPLE_POS_PER_TICKER = False  # recommended

QQQ_PATH = "data/qqq_dd52w.parquet"  # must contain columns: date, qqq_close

QQQ_DD_PATH = "data/qqq_dd52w.parquet"
TP_DD_BUCKETS = [
    (-0.30, 0.40),  # dd < -30% => reduced but still active
    (-0.20, 0.70),  # -30%..-20% => moderate reduction
    (-0.10, 0.90),  # -20%..-10% => very light reduction
    ( 1.00, 1.00),  # >= -10% => full size
]
# If True: reallocate the reduced TP budget to other strats (usually MR).
# If False: keep it in cash (recommended first run).
RENORMALIZE_AFTER_TP_SCALE = False

# TP regime gating
TP_GATE_MODE = "reduce"      # "off" or "reduce"
TP_RISKOFF_SCALE = 0.25   # used only if TP_GATE_MODE == "reduce"

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
    # stricter oversold requirement in risk-off
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


def build_qqq_regime_map(qqq_path: str) -> dict:
    q = pd.read_parquet(qqq_path).copy()
    q["date"] = pd.to_datetime(q["date"])
    if "qqq_close" not in q.columns:
        raise RuntimeError(f"QQQ file missing 'qqq_close'. Columns: {q.columns.tolist()}")

    q = q.sort_values("date").reset_index(drop=True)
    q["qqq_sma200"] = q["qqq_close"].rolling(200, min_periods=200).mean()
    q["risk_on"] = q["qqq_close"] > q["qqq_sma200"]

    # Map by normalized date
    return {pd.Timestamp(d).normalize(): bool(v) for d, v in zip(q["date"], q["risk_on"])}

def build_qqq_dd_map(path: str) -> dict:
    import pandas as pd
    q = pd.read_parquet(path).copy()
    q["date"] = pd.to_datetime(q["date"]).dt.normalize()
    if "qqq_dd_from_52w_high" not in q.columns:
        raise RuntimeError(f"qqq_dd52w missing qqq_dd_from_52w_high. Have: {q.columns.tolist()}")
    return dict(zip(q["date"], q["qqq_dd_from_52w_high"]))


def tp_scale_from_dd(dd: float) -> float:
    # dd is negative in drawdowns (e.g., -0.18 means -18%)
    if dd is None:
        return 1.0
    for cutoff, scale in TP_DD_BUCKETS:
        if dd < cutoff:
            return scale
    return 1.0

throttled_days_2025 = 0
total_pick_days_2025 = 0
# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)
    df = df.groupby("ticker", sort=False, group_keys=False).apply(ensure_features)

    START_DATE = "2015-10-16"
    END_DATE   = "2025-12-22"   # lock this for research comparability

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
    qqq_dd_map = build_qqq_dd_map(QQQ_DD_PATH)


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

        risk_on = qqq_risk_on.get(day.normalize(), True)

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
        for tkr in common:
            if (not ALLOW_MULTIPLE_POS_PER_TICKER) and (tkr in held_tickers):
                continue
            row = today.loc[tkr]
            atr = float(row.atr14_pct) if np.isfinite(row.atr14_pct) else np.nan
            if not np.isfinite(atr) or atr <= 0:
                continue

            if tp_signal(row):
                tp_cands.append((tkr, tp_rank(row), atr))
            if mr_signal(row, risk_on):
                mr_cands.append((tkr, mr_rank(row), atr))
            if eb_signal(row):
                eb_cands.append((tkr, eb_rank(row), atr))

            if not risk_on and row.ret3d <= -0.07 and row.ret3d > -0.09:
                print(day.date(), "MR blocked by risk-off gate", row.ret3d)

        tp_cands.sort(key=lambda x: x[1], reverse=True)
        mr_cands.sort(key=lambda x: x[1], reverse=True)
        eb_cands.sort(key=lambda x: x[1], reverse=True)

        tp_pick = tp_cands[:avail["TP_v2"]]
        mr_pick = mr_cands[:avail["MR_1A"]]
        eb_pick = eb_cands[:avail["EB_1A"]]

        # allocate budgets based on CURRENT equity (not cash), but spend from cash
        eq_now = eq
    
        # --- compute TP dd-scale once per day ---
        i = date_to_i[day]
        if i < len(dates) - 1:
            entry_day = pd.Timestamp(dates[i + 1]).normalize()
        else:
            entry_day = pd.Timestamp(day).normalize()

        dd_entry = qqq_dd_map.get(entry_day, None)
        tp_dd_scale = tp_scale_from_dd(dd_entry)
        if day.year == 2022 and day.month in (1, 6, 10) and day.day <= 5:
            print(
                f"[SANITY] {day.date()} -> entry {entry_day.date()} | "
                f"qqq_dd={dd_entry} | tp_scale={tp_dd_scale}"
            )

        # Optional: first-run debug
        # if day.year == 2022 and day.month in (1, 6, 12) and day.day == 3:
        #     print(f"{day.date()} dd={dd_today} tp_dd_scale={tp_dd_scale}")

        STRATS = ["TP_v2", "MR_1A", "EB_1A"]
        budgets = {s: 0.0 for s in STRATS}

        # If you want renormalization, we build effective weights here
        w_tp = CAP_W["TP_v2"] * tp_dd_scale
        w_mr = CAP_W["MR_1A"]
        w_eb = CAP_W.get("EB_1A", 0.0)

        if RENORMALIZE_AFTER_TP_SCALE:
            w_sum = w_tp + w_mr + w_eb
            if w_sum > 0:
                w_tp /= w_sum
                w_mr /= w_sum
                w_eb /= w_sum

        EFF_W = {"TP_v2": w_tp, "MR_1A": w_mr, "EB_1A": w_eb}

        for strat, picks in [("TP_v2", tp_pick), ("MR_1A", mr_pick), ("EB_1A", eb_pick)]:
            if len(picks) == 0:
                continue

            base = eq_now * CAP_W[strat]

            # TP gate / scale
            if strat == "TP_v2":
                if not risk_on:
                    if TP_GATE_MODE == "off":
                        base = 0.0
                    elif TP_GATE_MODE == "reduce":
                        base *= float(TP_RISKOFF_SCALE)
                    else:
                        raise RuntimeError(f"Unknown TP_GATE_MODE={TP_GATE_MODE}")
                else:
                    base *= float(tp_dd_scale)

            # MR dd-scale (new)
            if strat == "MR_1A":
                base *= float(tp_dd_scale)

            budgets[strat] = base

            if strat == "MR_1A" and day.year == 2022 and day.month in (1,6,10) and len(picks):
                print(f"[MR_BUDGET] {day.date()} tp_dd_scale={tp_dd_scale:.2f} base={base:.2f} picks={len(picks)}")


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
    print(f"[2025] throttled pick-days: {throttled_days_2025}/{total_pick_days_2025}")

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
