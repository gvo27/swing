import subprocess
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# ---------- CONFIG ----------
SCRIPTS = [
    "01_build_dataset_sp100.py",
    "08_build_market_regime_qqq.py",
    "15_build_qqq_dd52w.py",
    "v1_1_portfolio_tp_off_riskoff.py",
]

REQUIRED_OUTPUTS = {
    "sp100_daily_features.parquet": "data/sp100_daily_features.parquet",
    "qqq_regime.parquet": "data/qqq_regime.parquet",
    "qqq_dd52w.parquet": "data/qqq_dd52w.parquet",
}

# ---------- HELPERS ----------
def run_script(script_name: str):
    print(f"\n=== Running {script_name} ===")
    result = subprocess.run(
        [sys.executable, script_name],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed")

def check_file(path: str):
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Missing required output: {path}")
    return p

def sanity_check_parquet(path: str, date_col="date"):
    df = pd.read_parquet(path)
    if date_col not in df.columns:
        raise RuntimeError(f"{path} missing '{date_col}' column")

    last_date = pd.to_datetime(df[date_col]).max()
    print(f"{path} last date: {last_date.date()}")

    # Allow same-day or most recent trading day
    if (datetime.now().date() - last_date.date()).days > 3:
        raise RuntimeError(f"{path} looks stale")

# ---------- MAIN ----------
def main():
    print("========================================")
    print("RUN DAILY PIPELINE")
    print("========================================")

    # 1) Run build scripts
    for script in SCRIPTS[:-1]:
        run_script(script)

    # 2) Verify required outputs
    print("\n=== Verifying outputs ===")
    for name, path in REQUIRED_OUTPUTS.items():
        p = check_file(path)
        sanity_check_parquet(path)

    # 3) Run live order generator
    print("\n=== Generating tomorrow's orders ===")
    run_script(SCRIPTS[-1])

    print("\n✅ DAILY PIPELINE COMPLETE")

if __name__ == "__main__":
    main()
