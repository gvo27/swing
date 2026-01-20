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
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
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
    if df.empty:
        return df
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df["ticker"] = ticker
    return df.reset_index()

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
    print(f"Loaded {len(tickers)} tickers from S&P 100")

    frames = []
    for t in tqdm(tickers, desc="Downloading"):
        df = fetch_daily(t, CFG.start, CFG.end)
        if df.empty:
            print(f"WARNING: no data for {t}")
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("No data downloaded. Check internet / yfinance / tickers.")

    raw = pd.concat(frames, ignore_index=True)
    feat = add_features(raw)

    out_path = os.path.join(CFG.out_dir, CFG.out_file)
    feat.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}  rows={len(feat):,}  tickers={feat['ticker'].nunique()}")

if __name__ == "__main__":
    main()
