import pandas as pd
import os


# === CONFIG ===
MAX_DD = 3000
TARGET = 3000
TOLERANCE = 0.005   # 0.005 allows up to 0.5% blowups
MAX_SIZE = 10.0     # max size to search up to
PRECISION = 0.01    # size precision for binary search 0.01 is usually more than enough
MAX_RUNS = None   # max number of runs to simulate during search (None = all)

# --- Drawdown options ---
USE_TRAILING_DD = True

# --- Optional date filter ---

# START_DATE = "2025-05-01"          # set to None to disable filtering "YYYY-MM-DD"
# END_DATE = "2020-02-29"             # set to None to disable filtering "YYYY-MM-DD"
START_DATE = None
END_DATE = None

# --- Input CSV file ---
# input_file = "csvs/all_times.csv"
input_file = "csvs/premarket_only.csv"
# input_file = "csvs/top_times_only.csv"


def simulate_blowup_rate(df, size, target, max_dd, use_trailing=True, max_runs=None):
    """Quick simulation that returns blowup rate for given size."""
    n = len(df) if max_runs is None else min(len(df), max_runs)
    blowups = 0
    resolved = 0

    for start_idx in range(n):
        cumulative = 0
        peak = 0
        for i in range(start_idx, len(df)):
            pnl_today = df.loc[i, "P/L (Net)"] * size
            cumulative += pnl_today
            if use_trailing:
                peak = max(peak, cumulative)
                if cumulative < peak - max_dd:
                    blowups += 1
                    break
            else:
                if cumulative <= -max_dd:
                    blowups += 1
                    break
            if cumulative >= target:
                break
        resolved += 1
    return blowups / resolved if resolved else 0.0


def find_safe_size(df, target, max_dd, tol, max_size, precision, max_runs):
    """
    Binary search for largest size that keeps blowups <= tol (e.g., 0.005 = 0.5%).
    Returns: safe_size, blowup_rate
    """
    lo, hi = 0.01, max_size
    best = lo
    while hi - lo > precision:
        mid = (lo + hi) / 2
        blow_rate = simulate_blowup_rate(df, mid, target, max_dd, use_trailing=USE_TRAILING_DD, max_runs=max_runs)
        print(f"Testing size={mid:.2f} → blowups={blow_rate*100:.2f}%")
        if blow_rate <= tol:
            best = mid
            lo = mid
        else:
            hi = mid
    # Final re-check on full data
    final_rate = simulate_blowup_rate(df, best, target, max_dd, use_trailing=USE_TRAILING_DD, max_runs=None)
    return best, final_rate


# === Load and preprocess CSV ===
df = pd.read_csv(input_file, sep="\t")
input_filename = (os.path.basename(input_file)).replace(".csv", "")
print(f"📊 Loaded data from: {input_file}")

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# --- Clean numeric formatting ---
for col in ["P/L", "Net", "Hi", "Low", "Open", "Close"]:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(' ', '', regex=False)
            .str.replace(',', '.', regex=False)
            .astype(float)
        )

# --- Parse dates ---
df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")

# --- Create synthetic "P/L (Net)" column for compatibility ---
# We'll treat daily PnL as the change in Net from previous day
df["P/L (Net)"] = df["Net"].diff().fillna(df["P/L"].iloc[0] if "P/L" in df.columns else 0)

# --- Optional date filter ---
if START_DATE or END_DATE:
    if START_DATE:
        df = df[df["Date"] >= pd.to_datetime(START_DATE)]
    if END_DATE:
        df = df[df["Date"] <= pd.to_datetime(END_DATE)]
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"📅 Data filtered from {START_DATE or 'beginning'} to {END_DATE or 'end'}")
    print(f"Remaining rows: {len(df)}")


# === Find safe size ===
SAFE_SIZE, safe_rate = find_safe_size(df, TARGET, MAX_DD, TOLERANCE, MAX_SIZE, PRECISION, MAX_RUNS)
print(f"\n🔎 Safe size found: {SAFE_SIZE:.2f} contracts → blowups={safe_rate*100:.2f}%")
