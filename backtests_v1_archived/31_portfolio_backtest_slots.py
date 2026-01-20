import os
import numpy as np
import pandas as pd

PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"

# -----------------------------
# Portfolio config (YOU CHOSE)
# -----------------------------
MAX_TOTAL_SLOTS = 30
SLOTS = {"TP_v2": 15, "MR_1A": 10, "EB_1A": 5}

# Capital split (roughly proportional to slots; can change later)
CAP_W = {"TP_v2": 0.50, "MR_1A": 0.40, "EB_1A": 0.10}

# Locked strategy params
TP_HOLD = 20
TP_RET5D_MAX = -0.04
TP_ATR_MIN = 0.02

MR_HOLD = 10
MR_RET3D_MAX = -0.07
MR_ATR_MIN = 0.02

EB_HOLD = 10
EB_GAP_MIN = 0.04
EB_ATR_MIN = 0.02

# Slippage
SLIPPAGE_CENTS_PER_SHARE_PER_SIDE = 2.0
COMMISSION_PER_TRADE_USD = 0.0

START_CAPITAL = 1.0  # normalize to 1.0

# -----------------------------
# Helpers: fills & features
# -----------------------------
def apply_entry_slip(open_px: float) -> float:
    return float(open_px) + (SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0)

def apply_exit_slip(close_px: float) -> float:
    return float(close_px) - (SLIPPAGE_CENTS_PER_SHARE_PER_SIDE / 100.0)

def ensure_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    # MR features
    if "ret3d" not in g.columns:
        g["ret3d"] = g["close"] / g["close"].shift(3) - 1.0
    if "sma10" not in g.columns:
        g["sma10"] = g["close"].rolling(10, min_periods=10).mean()
    # EB features
    g["prev_close"] = g["close"].shift(1)
    g["gap"] = (g["open"] / g["prev_close"]) - 1.0
    g["day_ret"] = (g["close"] / g["open"]) - 1.0
    return g

# -----------------------------
# Signal (on close of day t), enter next open
# Ranking score: higher is better
# -----------------------------
def tp_signal_row(row) -> bool:
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
    # more negative ret5d is better => higher score
    return float(-row.ret5d)

def mr_signal_row(row) -> bool:
    if not np.isfinite(row.ret3d) or row.ret3d > MR_RET3D_MAX:
        return False
    if not np.isfinite(row.sma10) or row.close >= row.sma10:
        return False
    if not np.isfinite(row.atr14_pct) or row.atr14_pct < MR_ATR_MIN:
        return False
    return True

def mr_rank(row) -> float:
    return float(-row.ret3d)

def eb_signal_row(row) -> bool:
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
    # larger gap is better
    return float(row.gap)

# -----------------------------
# Position representation
# -----------------------------
class Position:
    __slots__ = ("strategy", "ticker", "entry_date", "exit_date", "shares", "entry_fill")

    def __init__(self, strategy, ticker, entry_date, exit_date, shares, entry_fill):
        self.strategy = strategy
        self.ticker = ticker
        self.entry_date = entry_date
        self.exit_date = exit_date
        self.shares = float(shares)
        self.entry_fill = float(entry_fill)

# -----------------------------
# Main portfolio backtest
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Build a per-date, per-ticker lookup (prices, atr, features)
    # We'll keep only needed columns for speed.
    need_cols = [
        "date","ticker","open","high","low","close",
        "atr14_pct","sma200","sma200_slope20","ret1d","ret5d",
    ]
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing column: {c}")

    # Add derived features
    df = df.groupby("ticker", sort=False, group_keys=False).apply(ensure_features)

    # For daily MTM we need close_t and close_{t-1} per ticker
    df["close_prev"] = df.groupby("ticker")["close"].shift(1)

    # Calendar
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    date_to_i = {d: i for i, d in enumerate(dates)}

    # Build a dict: (date -> frame of tickers that exist that date)
    # This avoids repeated filtering of the big df.
    frames_by_date = {}
    for d, g in df.groupby("date", sort=True):
        frames_by_date[pd.Timestamp(d)] = g.set_index("ticker", drop=False)

    # Portfolio state
    equity = START_CAPITAL
    equity_curve = []
    daily_ret = []
    holdings = []  # list[Position]

    # Track exposures per strategy for diagnostics
    active_counts = []
    active_by_strat = []

    for di, d in enumerate(dates):
        day = pd.Timestamp(d)
        today = frames_by_date.get(day)
        if today is None or len(today) == 0:
            continue

        # -----------------------------
        # 1) Mark-to-market from yesterday close to today close
        # -----------------------------
        pnl_today = 0.0
        pos_value_today = 0.0

        # We mark positions using close-to-close changes (shares * close)
        for pos in holdings:
            if pos.ticker not in today.index:
                continue
            close_today = float(today.loc[pos.ticker, "close"])
            close_prev = float(today.loc[pos.ticker, "close_prev"]) if np.isfinite(today.loc[pos.ticker, "close_prev"]) else np.nan
            if not np.isfinite(close_prev) or close_prev <= 0:
                continue
            # position value (approx) based on shares * close
            pos_value_today += pos.shares * close_today
            pnl_today += pos.shares * (close_today - close_prev)

        # Convert pnl to return on equity
        r_today = pnl_today / equity if equity > 0 else 0.0
        equity = equity * (1.0 + r_today)

        equity_curve.append({"date": day, "equity": equity})
        daily_ret.append({"date": day, "ret": r_today})

        # -----------------------------
        # 2) Exit positions whose exit_date == today (exit at today's close with slippage)
        #    We realize the difference between MTM close and slipped close on exit day as a small adjustment.
        # -----------------------------
        still = []
        exit_adjust_pnl = 0.0
        for pos in holdings:
            if pos.exit_date == day:
                if pos.ticker in today.index:
                    close_today = float(today.loc[pos.ticker, "close"])
                    exit_fill = apply_exit_slip(close_today)
                    # We already included close-to-close MTM, so adjust by (exit_fill - close_today)
                    exit_adjust_pnl += pos.shares * (exit_fill - close_today)
                    if COMMISSION_PER_TRADE_USD != 0.0:
                        exit_adjust_pnl -= COMMISSION_PER_TRADE_USD * 1.0
            else:
                still.append(pos)
        holdings = still

        if exit_adjust_pnl != 0.0:
            r_adj = exit_adjust_pnl / equity if equity > 0 else 0.0
            equity *= (1.0 + r_adj)
            equity_curve[-1]["equity"] = equity
            daily_ret[-1]["ret"] += r_adj

        # -----------------------------
        # 3) Enter new positions at NEXT day's open
        #    That means signals from today are queued for tomorrow open.
        # -----------------------------
        # If we're at the last day, can't enter.
        if di >= len(dates) - 1:
            continue

        next_day = dates[di + 1]
        tomorrow = frames_by_date.get(pd.Timestamp(next_day))
        if tomorrow is None or len(tomorrow) == 0:
            continue

        # Current active counts by strategy
        active_tp = sum(1 for p in holdings if p.strategy == "TP_v2")
        active_mr = sum(1 for p in holdings if p.strategy == "MR_1A")
        active_eb = sum(1 for p in holdings if p.strategy == "EB_1A")

        active_counts.append({"date": day, "active_total": len(holdings)})
        active_by_strat.append({"date": day, "TP_v2": active_tp, "MR_1A": active_mr, "EB_1A": active_eb})

        # Slots available
        avail = {
            "TP_v2": max(0, SLOTS["TP_v2"] - active_tp),
            "MR_1A": max(0, SLOTS["MR_1A"] - active_mr),
            "EB_1A": max(0, SLOTS["EB_1A"] - active_eb),
        }

        # Candidate generation from today's signals
        # Only consider tickers that exist tomorrow too (need open tomorrow)
        common_tickers = today.index.intersection(tomorrow.index)

        tp_cands, mr_cands, eb_cands = [], [], []
        for tkr in common_tickers:
            row = today.loc[tkr]

            if tp_signal_row(row):
                tp_cands.append((tkr, tp_rank(row), float(row.atr14_pct)))
            if mr_signal_row(row):
                mr_cands.append((tkr, mr_rank(row), float(row.atr14_pct)))
            if eb_signal_row(row):
                eb_cands.append((tkr, eb_rank(row), float(row.atr14_pct)))

        # Sort by rank (desc)
        tp_cands.sort(key=lambda x: x[1], reverse=True)
        mr_cands.sort(key=lambda x: x[1], reverse=True)
        eb_cands.sort(key=lambda x: x[1], reverse=True)

        tp_pick = tp_cands[:avail["TP_v2"]]
        mr_pick = mr_cands[:avail["MR_1A"]]
        eb_pick = eb_cands[:avail["EB_1A"]]

        # Budget per strategy (fraction of equity)
        # Allocate only to strategies that have picks today; otherwise leave cash unallocated.
        budgets = {}
        for strat, picks in [("TP_v2", tp_pick), ("MR_1A", mr_pick), ("EB_1A", eb_pick)]:
            budgets[strat] = equity * CAP_W[strat] if len(picks) > 0 else 0.0

        # For each strategy, size picks by inverse ATR and buy at tomorrow open (with slippage)
        for strat, picks, hold in [
            ("TP_v2", tp_pick, TP_HOLD),
            ("MR_1A", mr_pick, MR_HOLD),
            ("EB_1A", eb_pick, EB_HOLD),
        ]:
            if len(picks) == 0:
                continue
            budget = budgets[strat]
            if budget <= 0:
                continue

            inv_atr = np.array([1.0 / max(1e-6, p[2]) for p in picks], dtype=float)
            w = inv_atr / inv_atr.sum()

            for (tkr, _score, _atr), wi in zip(picks, w):
                open_tmr = float(tomorrow.loc[tkr, "open"])
                entry_fill = apply_entry_slip(open_tmr)
                if entry_fill <= 0 or not np.isfinite(entry_fill):
                    continue

                dollars = budget * float(wi)
                shares = dollars / entry_fill

                exit_idx = date_to_i[pd.Timestamp(next_day)] + hold
                if exit_idx >= len(dates):
                    continue
                exit_date = dates[exit_idx]

                holdings.append(Position(strat, tkr, pd.Timestamp(next_day), pd.Timestamp(exit_date), shares, entry_fill))

                # commission on entry
                if COMMISSION_PER_TRADE_USD != 0.0:
                    equity -= COMMISSION_PER_TRADE_USD / START_CAPITAL * START_CAPITAL  # tiny; kept simple

        # No need to renormalize equity after entries; we treat cash implicitly.

    eq = pd.DataFrame(equity_curve).set_index("date")
    rets = pd.DataFrame(daily_ret).set_index("date")

    # Metrics
    if len(rets) < 2:
        raise RuntimeError("Not enough daily returns produced.")

    total_ret = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    roll_max = eq["equity"].cummax()
    dd = eq["equity"] / roll_max - 1.0
    max_dd = float(dd.min())

    # Yearly returns
    eq_y = eq["equity"].resample("Y").last()
    eq_y_prev = eq_y.shift(1)
    yret = (eq_y / eq_y_prev - 1.0).dropna()
    yret.index = yret.index.year

    print("\n=== 30-slot Portfolio Backtest (TP/MR/EB) ===")
    print(f"Start: {eq.index[0].date()}   End: {eq.index[-1].date()}   Years: {years:.2f}")
    print(f"Total return: {total_ret:.2%}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Avg daily ret: {rets['ret'].mean():.5f}   Daily vol: {rets['ret'].std():.5f}")

    print("\nYearly returns:")
    print(yret.to_string())

    # Save outputs
    eq_path = os.path.join(OUT_DIR, "portfolio30_equity_curve.csv")
    dd_path = os.path.join(OUT_DIR, "portfolio30_drawdown.csv")
    rets_path = os.path.join(OUT_DIR, "portfolio30_daily_returns.csv")

    eq.to_csv(eq_path)
    dd.to_frame("drawdown").to_csv(dd_path)
    rets.to_csv(rets_path)

    print("\nSaved:")
    print(f"  {eq_path}")
    print(f"  {dd_path}")
    print(f"  {rets_path}")

if __name__ == "__main__":
    main()
