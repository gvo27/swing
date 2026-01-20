import os
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    start: str = "2015-01-01"
    end: Optional[str] = None
    out_dir: str = "data"
    out_file: str = "sp100_daily_features.parquet"


CFG = Config()


# ------------------
# Universe: S&P 100
# ------------------
def load_sp100_tickers():
    df = pd.read_csv("universes/sp100.csv")
    return df["ticker"].tolist()

# -----------------------------
# Indicators (no lookahead)
# -----------------------------
def rsi(close, period: int = 14) -> pd.Series:
    close = _as_series(close)

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(high, low, close, period: int = 14) -> pd.Series:
    high = _as_series(high)
    low  = _as_series(low)
    close = _as_series(close)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


# -----------------------------
# Data fetch + feature build
# -----------------------------
def fetch_daily(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    # If yfinance returns MultiIndex columns (sometimes happens), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if x not in (None, "")]).strip() for col in df.columns]

    # Normalize column names: lowercase + underscores
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # If duplicates exist, keep the first occurrence
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    df["ticker"] = ticker
    df = df.reset_index()

    return df

def _as_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    # If x is a DataFrame (multiple columns), take the first column
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    out = []


    for tkr, g in df.groupby("ticker", sort=False):
        g = g.copy()


        g["rsi14"] = rsi(g["close"], 14)
        g["sma50"] = sma(g["close"], 50)
        g["sma200"] = sma(g["close"], 200)

        g["atr14"] = atr(g["high"], g["low"], g["close"], 14)
        g["atr14_pct"] = g["atr14"] / g["close"]

        for n in [1, 5, 10, 20, 60]:
            g[f"ret{n}d"] = g["close"].pct_change(n)

        g["above_sma200"] = (g["close"] > g["sma200"]).astype("int")
        g["sma200_slope20"] = g["sma200"].pct_change(20)

        roll_high_252 = g["close"].rolling(252, min_periods=252).max()
        g["dd_from_52w_high"] = g["close"] / roll_high_252 - 1.0

        out.append(g)

    feat = pd.concat(out, ignore_index=True)
    # Drop rows without core indicators
    feat = feat.dropna(subset=["rsi14", "sma200", "atr14"])
    return feat

def main():
    os.makedirs(CFG.out_dir, exist_ok=True)

    tickers = load_sp100_tickers()
    tickers = [t for t in tickers if t != "WBA"]
    print(f"Loaded {len(tickers)} tickers (after filtering)")

    # --- Batch download (fast) ---
    df = yf.download(
        tickers=tickers,
        start=CFG.start,
        end=CFG.end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        raise RuntimeError("No data downloaded.")

    # df is typically MultiIndex columns: (ticker, field) when group_by="ticker"
    # But sometimes it's (field, ticker). We'll handle both.
    if not isinstance(df.columns, pd.MultiIndex):
        raise RuntimeError(f"Expected MultiIndex columns, got: {type(df.columns)}")

    lvl0 = df.columns.get_level_values(0)
    lvl1 = df.columns.get_level_values(1)

    # Decide which level is ticker by checking membership
    tickset = set(tickers)
    lvl0_is_ticker = sum(x in tickset for x in set(lvl0)) > sum(x in tickset for x in set(lvl1))

    if lvl0_is_ticker:
        # (ticker, field)
        long_df = df.stack(level=0).reset_index().rename(columns={"level_1": "ticker"})
        # columns now: date, ticker, Open, High, Low, Close, Adj Close, Volume
    else:
        # (field, ticker)
        long_df = df.stack(level=1).reset_index().rename(columns={"level_1": "ticker"})
        # columns now: date, ticker, Open, High, Low, Close, Adj Close, Volume

    # Normalize column names
    long_df.columns = [str(c).strip().lower().replace(" ", "_") for c in long_df.columns]
    # Ensure expected names
    # yfinance uses adj_close sometimes as 'adj_close' already after normalization
    # Drop rows missing core OHLC
    needed = {"open", "high", "low", "close"}
    missing = needed - set(long_df.columns)
    if missing:
        raise RuntimeError(f"Missing columns after reshape: {missing}. Columns are: {long_df.columns.tolist()}")

    # Drop tickers with no data (like WBA)
    long_df = long_df.dropna(subset=["close"]).copy()

    # Quick sanity
    print("Sanity columns:", long_df.columns.tolist())
    print("Tickers with data:", long_df["ticker"].nunique())

    feat = add_features(long_df)

    out_path = os.path.join(CFG.out_dir, CFG.out_file)
    feat.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}  rows={len(feat):,}  tickers={feat['ticker'].nunique()}")


if __name__ == "__main__":
    main()
