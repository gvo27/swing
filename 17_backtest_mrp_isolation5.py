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

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_pos"] = (df["close"] - df["low"]) / rng

    # Compute forward returns per ticker
    df["fwd_ret"] = df.groupby("ticker", group_keys=False).apply(lambda g: forward_return(g, HOLD_DAYS))
    df = df[df["fwd_ret"].notna()].copy()

    import warnings
    warnings.filterwarnings("ignore")  # optional: silences pandas warnings

    RET1D_LIST = [-0.03, -0.04, -0.05]
    RET5D_LIST = [-0.06, -0.08, -0.10]
    HOLD_LIST = [5, 7]

    results = []

    for hold in HOLD_LIST:
        df["fwd_ret"] = (df.groupby("ticker")["close"].shift(-hold) / df["close"] - 1.0)

        base = df[
            (df["market_stress"] == True) &
            (df["atr14_pct"] >= 0.02) &
            (df["fwd_ret"].notna())
        ].copy()

        base["year"] = base["date"].dt.year

        for r1 in RET1D_LIST:
            for r5 in RET5D_LIST:
                sub = base[
                    (base["ret1d"] <= r1) &
                    (base["ret5d"] <= r5)
                ].copy()

                if len(sub) < 100:
                    continue

                y22 = sub[sub["year"] == 2022]
                if len(y22) < 50:
                    continue

                results.append({
                    "hold": hold,
                    "ret1d_max": r1,
                    "ret5d_max": r5,
                    "n_2022": int(len(y22)),
                    "mean_2022": float(y22["fwd_ret"].mean()),
                    "win_2022": float((y22["fwd_ret"] > 0).mean()),
                    "n_all": int(len(sub)),
                    "mean_all": float(sub["fwd_ret"].mean()),
                    "win_all": float((sub["fwd_ret"] > 0).mean()),
                })

    res = pd.DataFrame(results)

    print("\n=== Top MR-P configs by 2022 mean_ret (ret5d filter) ===")
    if res.empty:
        print("No configs met thresholds. Lower y22 threshold to 20 and n_all to 50 for discovery.")
    else:
        print(res.sort_values("mean_2022", ascending=False).head(15).to_string(index=False))
                
if __name__ == "__main__":
    main()