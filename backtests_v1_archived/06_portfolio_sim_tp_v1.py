import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
PARQUET_PATH = "data/sp100_daily_features.parquet"
OUT_DIR = "data"

HOLDS = [5, 10, 20]                 # side-by-side portfolios
MAX_POSITIONS_LIST = [5, 10, 16, 20]  # sweep this (change freely)
START_CAPITAL = 100_000.0           # arbitrary scale; metrics are scale-invariant

# TP_v1 signal params (locked from your sweeps)
RET5D_MAX = -0.04
REQUIRE_DOWN_DAY = True
REQUIRE_TREND = True

# Costs OFF for now (we’ll add next)
COST_BPS_ROUND_TRIP = 0.0           # set e.g. 10 for 0.10% round-trip later


# -----------------------------
# SIGNAL + TRADE CANDIDATES
# -----------------------------
def build_tp_v1_signal(g: pd.DataFrame) -> pd.Series:
    cond = pd.Series(True, index=g.index)

    if REQUIRE_TREND:
        cond &= (g["above_sma200"] == 1)
        cond &= g["sma200_slope20"].notna()
        cond &= (g["sma200_slope20"] > 0)

    cond &= g["ret5d"].notna()
    cond &= (g["ret5d"] <= RET5D_MAX)

    if REQUIRE_DOWN_DAY:
        cond &= g["ret1d"].notna()
        cond &= (g["ret1d"] <= 0)

    # Need next day's open for entry
    cond &= g["open"].shift(-1).notna()
    return cond


def make_trade_candidates(df: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    """
    Creates one row per signal with:
      entry_date = next trading day for that ticker
      entry_price = open on entry_date
      exit_date = date at t+hold_days for that ticker
      exit_price = close on exit_date
    """
    df = df.sort_values(["ticker", "date"]).copy()

    out = []
    for tkr, g in df.groupby("ticker", sort=False):
        g = g.copy()
        sig = build_tp_v1_signal(g)

        # map signal day -> entry (t+1 open) and exit (t+H close)
        g["entry_date"] = g["date"].shift(-1)
        g["entry_price"] = g["open"].shift(-1)

        g["exit_date"] = g["date"].shift(-hold_days)
        g["exit_price"] = g["close"].shift(-hold_days)

        cand = g.loc[sig, ["date","ticker","ret1d","ret5d","entry_date","entry_price","exit_date","exit_price"]].copy()
        cand = cand.dropna(subset=["entry_date","entry_price","exit_date","exit_price"])

        cand["hold_days"] = hold_days

        # Ranking: deeper pullback first (more negative ret5d)
        cand["rank_key"] = cand["ret5d"]

        out.append(cand)

    if not out:
        return pd.DataFrame()

    cands = pd.concat(out, ignore_index=True)
    return cands


# -----------------------------
# PORTFOLIO SIM
# -----------------------------
@dataclass
class Position:
    ticker: str
    shares: float
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    hold_days: int


def simulate_portfolio(
    price_close: pd.Series,  # MultiIndex (date,ticker) -> close
    candidates: pd.DataFrame,
    max_positions: int,
    start_capital: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Equal-weight, max_positions cap, one position per ticker.
    Entries executed at entry_date at entry_price.
    Exits executed at exit_date at exit_price.
    Equity marked daily using CLOSE.
    """
    # Unique dates across the whole dataset (from price series)
    all_dates = price_close.index.get_level_values(0).unique().sort_values()

    # Group candidates by entry_date for fast lookup
    candidates = candidates.sort_values(["entry_date", "rank_key"]).copy()
    by_entry_date: Dict[pd.Timestamp, pd.DataFrame] = {
        d: grp for d, grp in candidates.groupby("entry_date", sort=False)
    }

    cash = float(start_capital)
    positions: Dict[str, Position] = {}  # ticker -> Position

    trades_log = []

    # For exits, we need to know which positions exit on each date
    # We'll just check each day (max 100 positions, cheap).

    equity_curve = []

    for d in all_dates:
        d = pd.Timestamp(d)

        # 1) Exits (at close)
        to_close = [tkr for tkr, pos in positions.items() if pos.exit_date == d]
        for tkr in to_close:
            pos = positions.pop(tkr)
            proceeds = pos.shares * pos.exit_price

            # Costs (round trip applied on notional)
            if COST_BPS_ROUND_TRIP and COST_BPS_ROUND_TRIP > 0:
                proceeds *= (1.0 - COST_BPS_ROUND_TRIP / 10_000.0)

            cash += proceeds

            trades_log.append({
                "ticker": pos.ticker,
                "hold_days": pos.hold_days,
                "entry_date": pos.entry_date,
                "entry_price": pos.entry_price,
                "exit_date": pos.exit_date,
                "exit_price": pos.exit_price,
                "shares": pos.shares,
                "pnl": proceeds - (pos.shares * pos.entry_price),
                "ret": (pos.exit_price / pos.entry_price) - 1.0
            })

        # 2) Entries (at open) — only for candidates whose entry_date == d
        if d in by_entry_date:
            todays = by_entry_date[d]

            # Process in priority order: most negative ret5d first
            todays = todays.sort_values("rank_key", ascending=True)

            for _, row in todays.iterrows():
                if len(positions) >= max_positions:
                    break

                tkr = row["ticker"]
                if tkr in positions:
                    continue  # one position per ticker

                entry_price = float(row["entry_price"])
                exit_date = pd.Timestamp(row["exit_date"])
                exit_price = float(row["exit_price"])
                hold_days = int(row["hold_days"])

                # Compute current equity using CLOSE marks
                # (cash + sum(shares * close_today))
                pos_value = 0.0
                for ptkr, pos in positions.items():
                    px = price_close.get((d, ptkr), np.nan)
                    if not np.isnan(px):
                        pos_value += pos.shares * float(px)
                    else:
                        # If missing close, fallback to last known (rare); simplest: keep at entry
                        pos_value += pos.shares * pos.entry_price

                equity = cash + pos_value

                # Equal-weight target allocation
                target_notional = equity / max_positions
                alloc = min(target_notional, cash)

                if alloc <= 0:
                    continue

                shares = alloc / entry_price
                cash -= shares * entry_price

                positions[tkr] = Position(
                    ticker=tkr,
                    shares=shares,
                    entry_date=d,
                    entry_price=entry_price,
                    exit_date=exit_date,
                    exit_price=exit_price,
                    hold_days=hold_days
                )

        # 3) Mark-to-market equity at close
        pos_value = 0.0
        for tkr, pos in positions.items():
            px = price_close.get((d, tkr), np.nan)
            if not np.isnan(px):
                pos_value += pos.shares * float(px)
            else:
                pos_value += pos.shares * pos.entry_price

        equity = cash + pos_value
        equity_curve.append({"date": d, "equity": equity, "cash": cash, "positions": len(positions)})

    eq = pd.DataFrame(equity_curve)
    trades = pd.DataFrame(trades_log)

    # Summary stats
    if len(eq) > 1:
        eq["ret1d"] = eq["equity"].pct_change()
        total_ret = eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0

        # Max drawdown
        peak = eq["equity"].cummax()
        dd = eq["equity"] / peak - 1.0
        max_dd = dd.min()

        # CAGR (approx using trading days)
        n_days = len(eq)
        years = n_days / 252.0
        cagr = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan

        avg_pos = eq["positions"].mean()
    else:
        total_ret, max_dd, cagr, avg_pos = np.nan, np.nan, np.nan, np.nan

    summary = {
        "total_return": total_ret,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "avg_positions": avg_pos,
        "trades": int(len(trades)) if trades is not None else 0
    }

    return eq, trades, summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Price series for marking positions daily
    price_close = df.set_index(["date","ticker"])["close"].sort_index()

    all_summaries = []
    all_equity = []
    all_trades = []

    for H in HOLDS:
        cands = make_trade_candidates(df, hold_days=H)
        if cands.empty:
            print(f"No candidates for hold={H}")
            continue

        for mp in MAX_POSITIONS_LIST:
            eq, trades, summ = simulate_portfolio(
                price_close=price_close,
                candidates=cands,
                max_positions=mp,
                start_capital=START_CAPITAL
            )

            summ_row = {
                "strategy": "TP_v1",
                "hold_days": H,
                "max_positions": mp,
                **summ
            }
            all_summaries.append(summ_row)

            eq_out = eq.copy()
            eq_out["strategy"] = "TP_v1"
            eq_out["hold_days"] = H
            eq_out["max_positions"] = mp
            all_equity.append(eq_out)

            if trades is not None and not trades.empty:
                tr_out = trades.copy()
                tr_out["strategy"] = "TP_v1"
                tr_out["hold_days"] = H
                tr_out["max_positions"] = mp
                all_trades.append(tr_out)

            print(f"Done hold={H} max_positions={mp}  "
                  f"CAGR={summ['cagr']:.3%}  MaxDD={summ['max_drawdown']:.3%}  Trades={summ['trades']}")

    summary_df = pd.DataFrame(all_summaries).sort_values(["hold_days","max_positions"])
    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    summary_path = os.path.join(OUT_DIR, "tp_portfolio_summary.csv")
    equity_path = os.path.join(OUT_DIR, "tp_portfolio_equity_curves.csv")
    trades_path = os.path.join(OUT_DIR, "tp_portfolio_trades.csv")

    summary_df.to_csv(summary_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    print("\nSaved:")
    print(" ", summary_path)
    print(" ", equity_path)
    print(" ", trades_path)

    print("\nSummary (sorted):")
    print(summary_df[["hold_days","max_positions","cagr","max_drawdown","avg_positions","trades","total_return"]].to_string(index=False))


if __name__ == "__main__":
    main()
