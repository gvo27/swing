import copy
import subprocess
import sys
import tempfile
import textwrap

"""
This script simply runs the same cash-ledger portfolio engine twice
with different slot/capital configs and prints a comparison summary.

It assumes:
- 31b_portfolio_backtest_slots_cash.py exists and works
- We modify only SLOTS and CAP_W between runs
"""

ENGINE_FILE = "31b_portfolio_backtest_slots_cash3.py"

# -----------------------------
# Two portfolio configs
# -----------------------------
PORTFOLIOS = {
    "TP_MR_only": {
        "SLOTS": {"TP_v2": 18, "MR_1A": 12, "EB_1A": 0},
        "CAP_W": {"TP_v2": 0.60, "MR_1A": 0.40, "EB_1A": 0.00},
    },
    "TP_MR_EB": {
        "SLOTS": {"TP_v2": 15, "MR_1A": 10, "EB_1A": 5},
        "CAP_W": {"TP_v2": 0.50, "MR_1A": 0.40, "EB_1A": 0.10},
    },
}

# -----------------------------
# Helper: run engine with overrides
# -----------------------------
def run_with_overrides(label, slots, cap_w):
    with open(ENGINE_FILE, "r") as f:
        base = f.read()

    patched = base

    # crude but effective text replacement
    patched = patched.replace(
        'SLOTS = {"TP_v2": 15, "MR_1A": 10, "EB_1A": 5}',
        f"SLOTS = {slots}"
    )
    patched = patched.replace(
        'CAP_W = {"TP_v2": 0.50, "MR_1A": 0.40, "EB_1A": 0.10}',
        f"CAP_W = {cap_w}"
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(patched)
        tmp_path = tmp.name

    print(f"\n=== Running {label} ===")
    proc = subprocess.run(
        [sys.executable, tmp_path],
        capture_output=True,
        text=True
    )

    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"{label} failed")

    print(proc.stdout)
    return proc.stdout

# -----------------------------
# Parse metrics from output
# -----------------------------
def parse_metrics(output: str):
    out = {}
    for line in output.splitlines():
        if line.startswith("Total return:"):
            out["total_return"] = line.split(":")[1].strip()
        if line.startswith("CAGR:"):
            out["cagr"] = line.split(":")[1].strip()
        if line.startswith("Max Drawdown:"):
            out["max_dd"] = line.split(":")[1].strip()
    return out

# -----------------------------
# Main
# -----------------------------
def main():
    results = {}

    for label, cfg in PORTFOLIOS.items():
        out = run_with_overrides(label, cfg["SLOTS"], cfg["CAP_W"])
        results[label] = parse_metrics(out)

    print("\n=== ABLATION SUMMARY ===")
    print(f"{'Model':<12} {'Total Return':>15} {'CAGR':>10} {'Max DD':>12}")
    for label, r in results.items():
        print(
            f"{label:<12} "
            f"{r.get('total_return','?'):>15} "
            f"{r.get('cagr','?'):>10} "
            f"{r.get('max_dd','?'):>12}"
        )

if __name__ == "__main__":
    main()
