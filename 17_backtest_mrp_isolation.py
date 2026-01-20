import pandas as pd
import numpy as np

DATA_PATH = "data/sp100_daily_features.parquet"
STRESS_PATH = "data/market_stress.parquet"

RET1D_MAX = -0.03
ATR_MIN = 0.02
HOLD_DAYS = 7  # start here; we'll sweep 5/7/10 later

def forward_return(g: pd.DataFrame, hold: int) -> pd.Series:
    return g["close"].shift(-hold) / g["close"] - 1.0

def main():
    df = pd.read_parquet(DATA_PATH).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    stress = pd.read_parquet(STRESS_PATH).copy()
    stress["date"] = pd.to_datetime(stress["date"]).dt.normalize()

    df = df.merge(stress[["date", "market_stress"]], on="date", how="left")
    df["market_stress"] = df["market_stress"].fillna(False)

    # Compute forward returns per ticker
    df["fwd_ret"] = df.groupby("ticker", group_keys=False).apply(lambda g: forward_return(g, HOLD_DAYS))
    df = df[df["fwd_ret"].notna()].copy()

    # MR-P condition (stress-only)
    cond = (
        (df["market_stress"] == True) &
        (df["ret1d"] <= RET1D_MAX) &
        (df["atr14_pct"] >= ATR_MIN)
    )

    picks = df[cond].copy()
    if len(picks) == 0:
        print("No MR-P trades found with current settings.")
        return

    # Summary
    picks["year"] = picks["date"].dt.year
    summary = {
        "n": len(picks),
        "win_rate": float((picks["fwd_ret"] > 0).mean()),
        "mean_ret": float(picks["fwd_ret"].mean()),
        "median_ret": float(picks["fwd_ret"].median()),
        "p25": float(picks["fwd_ret"].quantile(0.25)),
        "p75": float(picks["fwd_ret"].quantile(0.75)),
    }
    print("\nMR-P Isolation (stress-only)")
    print(f"RET1D_MAX={RET1D_MAX}, ATR_MIN={ATR_MIN}, HOLD={HOLD_DAYS}")
    print(summary)

    by_year = (
        picks.groupby("year")["fwd_ret"]
        .agg(n="count", win_rate=lambda x: (x > 0).mean(), mean_ret="mean", median_ret="median")
        .reset_index()
        .sort_values("year")
    )
    print("\nBy year:")
    print(by_year.to_string(index=False))

if __name__ == "__main__":
    main()
