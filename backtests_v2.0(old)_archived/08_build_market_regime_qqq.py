# 08_build_market_regime_qqq.py
# Builds a simple, explicit 3-regime model for QQQ:
#   STRONG:   QQQ>SMA200 AND SMA50>SMA200 AND dd52w >= -0.12
#   RISK_OFF: QQQ<SMA200 OR dd52w <= -0.20
#   NEUTRAL:  otherwise
# Plus a 3-day confirmation filter to reduce whipsaw.

import os
import numpy as np
import pandas as pd


QQQ_IN_PATH = "data/qqq_dd52w.parquet"          # expects at least: date, qqq_close (or close/adj_close)
QQQ_OUT_PATH = "data/qqq_regime.parquet"

DD_STRONG = -0.12     # dd52w threshold to allow STRONG
DD_RISKOFF = -0.20    # dd52w threshold to force RISK_OFF
CONFIRM_DAYS = 3      # consecutive days required to confirm STRONG/RISK_OFF

VOL_WIN = 20          # realized vol window (std of daily returns)
DD_WIN = 252          # 52w window (trading days) for rolling high


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Handles MultiIndex/tuple columns like ("date","") or ("close","qqq")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in tup if str(x) not in ("", "None")]).strip("_")
            for tup in df.columns.to_list()
        ]
    else:
        df.columns = [
            ("_".join([str(x) for x in c if str(x) not in ("", "None")]).strip("_")
             if isinstance(c, tuple) else str(c))
            for c in df.columns
        ]
    return df


def pick_price_column(q: pd.DataFrame) -> str:
    # Prefer explicit qqq_close, else close/adj_close variants
    candidates = [
        "qqq_close",
        "close",
        "adj_close",
        "adjclose",
        "Adj Close",
        "Close",
        "('close',_'qqq')",
        "('adj_close',_'qqq')",
    ]
    for c in candidates:
        if c in q.columns:
            return c

    # Heuristic: anything containing "close" and "qqq"
    close_like = [c for c in q.columns if "close" in c.lower() and "qqq" in c.lower()]
    if close_like:
        return close_like[0]

    raise RuntimeError(f"Could not find a QQQ price column. Columns: {q.columns.tolist()}")


def confirm_state(cond: pd.Series, k: int) -> pd.Series:
    # True only if condition has been true for k consecutive days (including today)
    # Uses rolling sum on boolean.
    return cond.rolling(k, min_periods=k).sum().fillna(0).ge(k)


def main():
    if not os.path.exists(QQQ_IN_PATH):
        raise FileNotFoundError(f"Missing input: {QQQ_IN_PATH}")

    q = pd.read_parquet(QQQ_IN_PATH).copy()
    q = flatten_columns(q)

    # Date column detection
    if "date" not in q.columns:
        # common variant
        date_like = [c for c in q.columns if c.lower() == "date" or c.lower().endswith("date")]
        if date_like:
            q = q.rename(columns={date_like[0]: "date"})
        else:
            raise RuntimeError(f"Could not find date column. Columns: {q.columns.tolist()}")

    q["date"] = pd.to_datetime(q["date"])
    q = q.sort_values("date").reset_index(drop=True)

    px_col = pick_price_column(q)
    q = q.rename(columns={px_col: "qqq_close"})
    q["qqq_close"] = pd.to_numeric(q["qqq_close"], errors="coerce")

    if q["qqq_close"].isna().all():
        raise RuntimeError("qqq_close is all NaN after parsing. Check your input file.")

    # Indicators
    q["ret1d"] = q["qqq_close"].pct_change(1)
    q["vol20"] = q["ret1d"].rolling(VOL_WIN, min_periods=VOL_WIN).std()

    q["sma50"] = q["qqq_close"].rolling(50, min_periods=50).mean()
    q["sma200"] = q["qqq_close"].rolling(200, min_periods=200).mean()

    # 52w drawdown
    q["high52w"] = q["qqq_close"].rolling(DD_WIN, min_periods=DD_WIN).max()
    q["dd52w"] = (q["qqq_close"] / q["high52w"]) - 1.0

    # Core conditions
    q["risk_on"] = q["qqq_close"] > q["sma200"]
    q["risk_on_strong"] = (q["qqq_close"] > q["sma200"]) & (q["sma50"] > q["sma200"])

    q["cond_strong"] = (
        (q["qqq_close"] > q["sma200"]) &
        (q["sma50"] > q["sma200"]) &
        (q["dd52w"] >= DD_STRONG)
    )

    q["cond_riskoff"] = (
        (q["qqq_close"] < q["sma200"]) |
        (q["dd52w"] <= DD_RISKOFF)
    )

    # Raw regime (priority: riskoff first, then strong, else neutral)
    q["regime_raw"] = np.where(
        q["cond_riskoff"], "RISK_OFF",
        np.where(q["cond_strong"], "STRONG", "NEUTRAL")
    )

    # 3-day confirmation (only STRONG and RISK_OFF require confirmation)
    strong_conf = confirm_state(q["cond_strong"].astype(bool), CONFIRM_DAYS)
    riskoff_conf = confirm_state(q["cond_riskoff"].astype(bool), CONFIRM_DAYS)

    # Confirmed regime:
    # - If riskoff confirmed -> RISK_OFF
    # - Else if strong confirmed -> STRONG
    # - Else NEUTRAL
    q["regime"] = np.where(
        riskoff_conf, "RISK_OFF",
        np.where(strong_conf, "STRONG", "NEUTRAL")
    )

    # Helpful: a single boolean "risk_on_confirmed" you can use for gating TP
    q["risk_on_confirmed"] = q["regime"].isin(["STRONG", "NEUTRAL"]) & (~riskoff_conf)

    out_cols = [
        "date", "qqq_close",
        "ret1d", "vol20",
        "sma50", "sma200",
        "high52w", "dd52w",
        "risk_on", "risk_on_strong",
        "cond_strong", "cond_riskoff",
        "regime_raw", "regime", "risk_on_confirmed"
    ]

    out = q[out_cols].copy()

    os.makedirs(os.path.dirname(QQQ_OUT_PATH), exist_ok=True)
    out.to_parquet(QQQ_OUT_PATH, index=False)

    # Console summary
    print(f"Wrote: {QQQ_OUT_PATH}")
    print("Regime counts:")
    print(out["regime"].value_counts(dropna=False).to_string())
    print("\nLatest row:")
    print(out.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
