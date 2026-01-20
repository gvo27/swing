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

    DD52W_LIST = [-0.05, -0.10, -0.15]
    CLOSE_POS_LIST = [0.55, 0.60, 0.65]
    HOLD_LIST = [5, 7]

    results = []

    for hold in HOLD_LIST:
        # Compute forward return ONCE per hold
        fwd = (
            df.groupby("ticker")["close"]
              .shift(-hold) / df["close"] - 1.0
        )
        df["fwd_ret"] = fwd

        # Base filter (shared across all combos)
        base = df[
            (df["market_stress"] == True) &
            (df["ret1d"] <= -0.03) &
            (df["atr14_pct"] >= 0.02) &
            (df["fwd_ret"].notna())
        ].copy()

        if base.empty:
            print(f"[hold={hold}] base is empty (no rows pass stress+ret1d+atr).")
            continue

        base["year"] = base["date"].dt.year
        base_2022 = base[base["year"] == 2022]
  
        print(f"[hold={hold}] base n_all={len(base)}, n_2022={len(base[base['year']==2022])}")

        # how many survive dd52w alone (no close_pos)
        for dd52w in DD52W_LIST:
            tmp = base[base["dd_from_52w_high"] <= dd52w]
            print(f"  dd52w<={dd52w}: n_all={len(tmp)}, n_2022={len(tmp[tmp['year']==2022])}")

        # how many survive close_pos alone (no dd52w)
        for cp in CLOSE_POS_LIST:
            tmp = base[base["close_pos"] >= cp]
            print(f"  close_pos>={cp}: n_all={len(tmp)}, n_2022={len(tmp[tmp['year']==2022])}")

        for dd52w in DD52W_LIST:
            for cp in CLOSE_POS_LIST:
                sub = base[
                    (base["dd_from_52w_high"] <= dd52w) &
                    (base["close_pos"] >= cp)
                ].copy()

                # minimum sample thresholds
                if len(sub) < 50:
                    continue

                y22 = sub[sub["year"] == 2022]
                if len(y22) < 20:
                    continue

                results.append({
                    "hold": hold,
                    "dd52w": dd52w,
                    "close_pos": cp,
                    "n_2022": int(len(y22)),
                    "mean_2022": float(y22["fwd_ret"].mean()),
                    "win_2022": float((y22["fwd_ret"] > 0).mean()),
                    "n_all": int(len(sub)),
                    "mean_all": float(sub["fwd_ret"].mean()),
                    "win_all": float((sub["fwd_ret"] > 0).mean()),
                })

    res = pd.DataFrame(results)

    print("\n=== Top MR-P configs by 2022 mean_ret ===")
    if res.empty:
        print("No configs met thresholds. Try lowering constraints:")
        print("- y22 min trades: 50 -> 20")
        print("- all min trades: 100 -> 50")
        print("- close_pos: 0.55 only")
        print("- dd52w: -0.10 only")
    else:
        print(
            res.sort_values("mean_2022", ascending=False)
               .head(15)
               .to_string(index=False)
        )

if __name__ == "__main__":
    main()
