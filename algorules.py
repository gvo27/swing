#Stock List: SP500
#IMPORTANT: all features are EOD-t, labels are t+N.

RET_WINDOWS = [1,2,3,5,7,10,15,20,30,60,120]

RET_THRESHOLDS = {
    "ret1d_max": [-0.01,-0.02,-0.03,-0.04,-0.05,-0.06,-0.08],
    "ret3d_max": [-0.03,-0.05,-0.07,-0.09,-0.12,-0.15],
    "ret5d_max": [-0.02,-0.04,-0.06,-0.08,-0.10,-0.12,-0.15],
    "ret10d_max": [-0.05,-0.08,-0.10,-0.15],
    "ret20d_min": [-0.10,-0.05,-0.02,0.00,0.02,0.05],
}

ATR_WINDOWS = [5, 14, 20]

ATR_THRESHOLDS = {
    "atr_min": [0.01,0.015,0.02,0.025,0.03,0.035,0.04],
    "atr_max": [0.10,0.15,0.20],  # optional upper cap
}

VOL_THRESHOLDS = {
    "vol10_min": [0.01,0.015,0.02],
    "vol20_min": [0.01,0.015,0.02],
    "vol60_min": [0.01,0.015,0.02],
}

SMA_WINDOWS = [5,10,20,50,100,200]

TREND_THRESHOLDS = {
    "above_sma10_min": [-0.10,-0.05,0.00,0.02,0.05],
    "above_sma20_min": [-0.10,-0.05,0.00,0.02,0.05],
    "above_sma50_min": [-0.10,-0.05,0.00,0.02,0.05],
    "above_sma200_min": [-0.10,-0.05,0.00,0.02,0.05,0.10],

    "sma200_slope20_min": [-0.02,0.00,0.01,0.02,0.05],
}

DRAWDOWN_THRESHOLDS = {
    "dd20_max": [-0.03,-0.05,-0.08,-0.12,-0.15],
    "dd52w_max": [-0.05,-0.10,-0.15,-0.20,-0.25,-0.30],
}

RANGE_POSITION_THRESHOLDS = {
    "close_pos_20d_min": [0.40,0.45,0.50,0.55,0.60,0.65],
    "close_pos_252d_min": [0.40,0.45,0.50,0.55,0.60,0.65],
}

GAP_THRESHOLDS = {
    "gap_min": [0.01,0.02,0.03,0.04,0.05,0.07,0.10],
    "gap_max": [-0.01,-0.02,-0.03,-0.05],  # downside gaps
}

INTRADAY_THRESHOLDS = {
    "day_ret_min": [0.00,0.01,0.02],
    "range_pct_min": [0.01,0.02,0.03],
}

MARKET_FEATURES = {
    "spy_ret5d": [-0.10,-0.05,-0.02,0.00],
    "spy_ret20d": [-0.15,-0.10,-0.05,0.00],
    "spy_dd52w": [-0.10,-0.15,-0.20,-0.25],
}

import numpy as np
import pandas as pd

# -----------------------------
# Helpers
# -----------------------------
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    df = df.sort_index()
    return df


def add_return_features(df: pd.DataFrame, close_col="close", windows=(1, 3, 5, 10, 20)) -> pd.DataFrame:
    """
    Adds: ret{w}d  (simple returns over w trading days)
    Example: ret1d, ret3d, ret5d, ret10d, ret20d
    """
    out = df.copy()
    c = out[close_col].astype(float)

    for w in windows:
        out[f"ret{w}d"] = c.pct_change(w)
    return out


def add_intraday_features(df: pd.DataFrame,
                          open_col="open",
                          high_col="high",
                          low_col="low",
                          close_col="close") -> pd.DataFrame:
    """
    Adds:
      day_ret      = close/open - 1
      range_pct    = (high-low)/open   (you can change denominator to close if you prefer)
    """
    out = df.copy()
    o = out[open_col].astype(float)
    h = out[high_col].astype(float)
    l = out[low_col].astype(float)
    c = out[close_col].astype(float)

    out["day_ret"] = (c / o) - 1.0
    out["range_pct"] = (h - l) / o
    return out


def add_gap_features(df: pd.DataFrame,
                     open_col="open",
                     close_col="close") -> pd.DataFrame:
    """
    Adds:
      gap = open / prior_close - 1
    """
    out = df.copy()
    o = out[open_col].astype(float)
    c = out[close_col].astype(float)
    out["gap"] = (o / c.shift(1)) - 1.0
    return out


def add_volatility_features(df: pd.DataFrame, close_col="close", windows=(10, 20, 60)) -> pd.DataFrame:
    """
    Adds:
      vol{w} = rolling std of daily returns over w days
    Note: this is daily-return std (NOT annualized).
    """
    out = df.copy()
    c = out[close_col].astype(float)
    r1 = c.pct_change(1)

    for w in windows:
        out[f"vol{w}"] = r1.rolling(w, min_periods=w).std()
    return out


def add_sma_features(df: pd.DataFrame, close_col="close", windows=(5, 10, 20, 50, 100, 200)) -> pd.DataFrame:
    """
    Adds:
      sma{w}
      above_sma{w} = close/sma{w} - 1  (distance above/below SMA)
    """
    out = df.copy()
    c = out[close_col].astype(float)

    for w in windows:
        sma = c.rolling(w, min_periods=w).mean()
        out[f"sma{w}"] = sma
        out[f"above_sma{w}"] = (c / sma) - 1.0
    return out


def add_sma_slope_feature(df: pd.DataFrame, sma_window=200, slope_window=20) -> pd.DataFrame:
    """
    Adds:
      sma{S}_slope{K} = slope over K days of sma{S}, expressed as pct change over K
                       = sma200 / sma200.shift(20) - 1
    """
    out = df.copy()
    sma_col = f"sma{sma_window}"
    if sma_col not in out.columns:
        out = add_sma_features(out, windows=(sma_window,))

    sma = out[sma_col].astype(float)
    out[f"sma{sma_window}_slope{slope_window}"] = (sma / sma.shift(slope_window)) - 1.0
    return out


def add_drawdown_features(df: pd.DataFrame, close_col="close", windows=(20, 252)) -> pd.DataFrame:
    """
    Adds:
      dd{w} = close / rolling_max(close, w) - 1
    Examples:
      dd20, dd252 (proxy for 52w in trading days)
    """
    out = df.copy()
    c = out[close_col].astype(float)

    for w in windows:
        roll_max = c.rolling(w, min_periods=w).max()
        out[f"dd{w}"] = (c / roll_max) - 1.0
    return out


def add_range_position_features(df: pd.DataFrame, close_col="close", windows=(20, 252)) -> pd.DataFrame:
    """
    Adds:
      close_pos_{w}d in [0,1] where:
        (close - rolling_min) / (rolling_max - rolling_min)
    If max==min (flat), result is NaN for that day.
    """
    out = df.copy()
    c = out[close_col].astype(float)

    for w in windows:
        lo = c.rolling(w, min_periods=w).min()
        hi = c.rolling(w, min_periods=w).max()
        denom = (hi - lo).replace(0.0, np.nan)
        out[f"close_pos_{w}d"] = (c - lo) / denom
    return out


def add_atr_features(df: pd.DataFrame,
                     high_col="high",
                     low_col="low",
                     close_col="close",
                     windows=(5, 14, 20),
                     pct_denom="close") -> pd.DataFrame:
    """
    Adds:
      atr{w}        (absolute ATR)
      atr{w}_pct    ATR as percent of denom (close by default)
    True Range (Wilder):
      TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    ATR here uses Wilder's smoothing via ewm(alpha=1/w, adjust=False).
    If you prefer SMA ATR, swap the ewm for rolling(w).mean().
    """
    out = df.copy()
    h = out[high_col].astype(float)
    l = out[low_col].astype(float)
    c = out[close_col].astype(float)

    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)

    for w in windows:
        atr = tr.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean()
        out[f"atr{w}"] = atr

        if pct_denom == "close":
            denom = c
        elif pct_denom == "prev_close":
            denom = prev_c
        elif pct_denom == "open":
            denom = out["open"].astype(float)
        else:
            raise ValueError("pct_denom must be one of: 'close', 'prev_close', 'open'")

        out[f"atr{w}_pct"] = atr / denom
    return out


def compute_single_ticker_features(
    ohlcv: pd.DataFrame,
    *,
    ret_windows=(1, 3, 5, 10, 20),
    atr_windows=(5, 14, 20),
    vol_windows=(10, 20, 60),
    sma_windows=(5, 10, 20, 50, 100, 200),
    dd_windows=(20, 252),
    range_pos_windows=(20, 252),
) -> pd.DataFrame:
    """
    Convenience wrapper. Input must have columns:
      open, high, low, close (volume optional)
    Index should be datetime OR include a 'date' column.
    """
    df = _ensure_datetime_index(ohlcv)

    df = add_return_features(df, windows=ret_windows)
    df = add_intraday_features(df)
    df = add_gap_features(df)
    df = add_volatility_features(df, windows=vol_windows)
    df = add_sma_features(df, windows=sma_windows)
    df = add_sma_slope_feature(df, sma_window=200, slope_window=20)  # your sma200_slope20
    df = add_drawdown_features(df, windows=dd_windows)
    df = add_range_position_features(df, windows=range_pos_windows)
    df = add_atr_features(df, windows=atr_windows, pct_denom="close")

    return df
