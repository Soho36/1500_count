import pandas as pd
import numpy as np

# (assume df and your other config variables are already defined above)
# I'll reuse TARGET, MAX_DD, SIZE, CONTRACT_STEP, USE_DYNAMIC_LOT, etc.
# If not, define them before calling simulate.

# --- Load & clean data ---
df = pd.read_csv("csvs/all_times.csv", sep="\t")
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Increase display width

# Clean number formatting
df['P/L (Net)'] = (
    df['P/L (Net)']
    .astype(str)
    .str.replace(' ', '', regex=False)
    .str.replace(',', '.', regex=False)
    .astype(float)
)

# === CONFIG ===
MAX_DD = 1500               # maximum drawdown allowed before "blowup"
TARGET = 1500               # profit target per run

SIZE = 1                    # static lot size (if not using dynamic)
CONTRACT_STEP = 500         # add/remove 1 contract per $500 gain/loss
USE_DYNAMIC_LOT = False     # 🔄 switch: True = dynamic lot, False = static
USE_TRAILING_DD = False      # 🔁 switch: True = trailing DD, False = static DD
SAVE_CONTRACT_LOG = True    # save detailed per-day info for first N runs
MAX_RUNS_TO_LOG = 1000      # limit detailed log to first N runs
# ==============

# results = []
# detailed_log = []


def simulate(df, use_dynamic_lot, use_trailing_dd, target, max_dd, size, contract_step, max_runs_to_log=1000, save_log=False):
    results = []
    detailed_log = []

    for start_idx in range(len(df)):
        cumulative_pnl = 0.0
        min_cumulative_pnl = 0.0
        days = 0
        reached = False
        blown = False

        peak_pnl = 0.0
        trailing_floor = -max_dd

        contracts = size if not use_dynamic_lot else 1
        contract_history = []

        for i in range(start_idx, len(df)):
            contract_history.append(contracts)
            pnl_today = df.loc[i, 'P/L (Net)'] * (contracts if use_dynamic_lot else size)
            cumulative_pnl += pnl_today
            min_cumulative_pnl = min(min_cumulative_pnl, cumulative_pnl)
            days += 1

            if save_log and start_idx < max_runs_to_log:
                detailed_log.append({
                    "Run_Start": df.loc[start_idx, 'Date'],
                    "Date": df.loc[i, 'Date'],
                    "Contracts": contracts,
                    "PnL_Today": round(pnl_today, 2),
                    "Cumulative_PnL": round(cumulative_pnl, 2),
                    "Peak_PnL": round(peak_pnl, 2),
                    "Trailing_Floor": round(trailing_floor, 2)
                })

            if use_dynamic_lot:
                contracts = max(1, 1 + int(cumulative_pnl // contract_step))

            if use_trailing_dd:
                peak_pnl = max(peak_pnl, cumulative_pnl)
                trailing_floor = peak_pnl - max_dd
                dd_breached = cumulative_pnl < trailing_floor
            else:
                dd_breached = cumulative_pnl <= -max_dd

            if dd_breached:
                results.append({
                    "Start_Date": df.loc[start_idx, 'Date'],
                    "Rows_to_+Target": None,
                    "Max_Drawdown": (peak_pnl - cumulative_pnl) if use_trailing_dd else abs(min_cumulative_pnl),
                    "Average_Contracts": sum(contract_history)/len(contract_history) if use_dynamic_lot else size,
                    "Minimum_Contracts": min(contract_history) if use_dynamic_lot else size,
                    "Maximum_Contracts": max(contract_history) if use_dynamic_lot else size,
                    "End_Date": df.loc[i, 'Date'],
                    "Blown": True
                })
                blown = True
                break

            if cumulative_pnl >= target:
                results.append({
                    "Start_Date": df.loc[start_idx, 'Date'],
                    "Rows_to_+Target": days,
                    "Max_Drawdown": abs(min_cumulative_pnl),
                    "Average_Contracts": sum(contract_history)/len(contract_history) if use_dynamic_lot else size,
                    "Minimum_Contracts": min(contract_history) if use_dynamic_lot else size,
                    "Maximum_Contracts": max(contract_history) if use_dynamic_lot else size,
                    "End_Date": df.loc[i, 'Date'],
                    "Blown": False
                })
                reached = True
                break

        if not reached and not blown:
            results.append({
                "Start_Date": df.loc[start_idx, 'Date'],
                "Rows_to_+Target": None,
                "Max_Drawdown": abs(min_cumulative_pnl),
                "Average_Contracts": sum(contract_history)/len(contract_history) if use_dynamic_lot else size,
                "Minimum_Contracts": min(contract_history) if use_dynamic_lot else size,
                "Maximum_Contracts": max(contract_history) if use_dynamic_lot else size,
                "End_Date": None,
                "Blown": False
            })

    return pd.DataFrame(results), pd.DataFrame(detailed_log)


# Run both modes (static DD and trailing DD) keeping other params equal
res_static, log_static = simulate(df, USE_DYNAMIC_LOT, False, TARGET, MAX_DD, SIZE, CONTRACT_STEP, MAX_RUNS_TO_LOG, SAVE_CONTRACT_LOG)
res_trail, log_trail = simulate(df, USE_DYNAMIC_LOT, True, TARGET, MAX_DD, SIZE, CONTRACT_STEP, MAX_RUNS_TO_LOG, SAVE_CONTRACT_LOG)


# Basic summary comparisons
def summary_stats(results_df):
    valid = results_df.dropna(subset=["Rows_to_+Target"])
    return {
        "total_runs": len(results_df),
        "valid_runs": len(valid),
        "mean_days": valid["Rows_to_+Target"].mean() if len(valid) else np.nan,
        "median_days": valid["Rows_to_+Target"].median() if len(valid) else np.nan,
        "mode_days": valid["Rows_to_+Target"].mode().values if len(valid) else np.array([]),
        "std_days": valid["Rows_to_+Target"].std() if len(valid) else np.nan
    }


# print("STATIC DD summary:", summary_stats(res_static))
# print("TRAILING DD summary:", summary_stats(res_trail))


stats_df = pd.DataFrame({
    "Static DD": summary_stats(res_static),
    "Trailing DD": summary_stats(res_trail)
})

print("\n=== COMPARISON TABLE ===")
print(stats_df)


# Find start dates that change outcome
merged = pd.merge(
    res_static[["Start_Date", "Rows_to_+Target", "Blown"]].rename(columns={"Rows_to_+Target":"Rows_static", "Blown":"Blown_static"}),
    res_trail[["Start_Date", "Rows_to_+Target", "Blown"]].rename(columns={"Rows_to_+Target":"Rows_trail", "Blown":"Blown_trail"}),
    on="Start_Date", how="outer"
)

# Cases where static succeeded but trailing blew (the intuitive 'lost after high' cases)
diff1 = merged[(merged["Rows_static"].notna()) & (merged["Blown_trail"] == True)]
print("\nRuns SUCCESS in static but BLOWN in trailing (sample):", len(diff1))
print(diff1.head(200))

# Cases where trailing succeeded but static blew (rare, but check)
diff2 = merged[(merged["Rows_trail"].notna()) & (merged["Blown_static"] == True)]
print("\nRuns SUCCESS in trailing but BLOWN in static (sample):", len(diff2))
print(diff2.head(200))

# Compare distributions directly
print("\nStatic valid days distribution value_counts (top 10):")
print(res_static["Rows_to_+Target"].value_counts().sort_index().head(20))
print("\nTrailing valid days distribution value_counts (top 20):")
print(res_trail["Rows_to_+Target"].value_counts().sort_index().head(20))
