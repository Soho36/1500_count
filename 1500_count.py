import pandas as pd

# --- Load & clean data ---
df = pd.read_csv("../1500 count/csvs/with_pnl.csv", sep="\t")
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
TARGET = 6000               # profit target per run
MAX_DD = 3000               # maximum drawdown allowed before "blowup"
SIZE = 2                    # static lot size (if not using dynamic)
CONTRACT_STEP = 500         # add/remove 1 contract per $500 gain/loss
USE_DYNAMIC_LOT = False     # 🔄 switch: True = dynamic lot, False = static
SAVE_CONTRACT_LOG = True    # <--- set to False to skip detailed log

MAX_RUNS_TO_LOG = 1000       # limit detailed log to first N runs

# ==============

results = []
detailed_log = []

# --- Loop through every possible starting date ---
for start_idx in range(len(df)):
    cumulative_pnl = 0
    min_cumulative_pnl = 0
    days = 0
    reached = False
    blown = False

    # --- reset trailing DD vars per run ---
    peak_pnl = 0
    trailing_floor = -MAX_DD

    # --- dynamic lot setup ---
    contracts = SIZE if not USE_DYNAMIC_LOT else 1
    contract_history = []

    for i in range(start_idx, len(df)):
        # Record contract size
        contract_history.append(contracts)

        # --- Apply today's PnL ---
        pnl_today = df.loc[i, 'P/L (Net)'] * (contracts if USE_DYNAMIC_LOT else SIZE)
        cumulative_pnl += pnl_today
        min_cumulative_pnl = min(min_cumulative_pnl, cumulative_pnl)
        days += 1

        # --- save per-day details ---
        if SAVE_CONTRACT_LOG and start_idx < MAX_RUNS_TO_LOG:
            detailed_log.append({
                "Run_Start": df.loc[start_idx, 'Date'],
                "Date": df.loc[i, 'Date'],
                "Contracts": contracts,
                "PnL_Today": round(pnl_today, 2),
                "Cumulative_PnL": round(cumulative_pnl, 2),
                "Peak_PnL": round(peak_pnl, 2),
                "Trailing_Floor": round(trailing_floor, 2)
            })

        # --- Update contract size dynamically (only if enabled) ---
        if USE_DYNAMIC_LOT:
            contracts = max(1, 1 + int(cumulative_pnl // CONTRACT_STEP))

        # --- Update trailing DD ---
        peak_pnl = max(peak_pnl, cumulative_pnl)
        trailing_floor = peak_pnl - MAX_DD

        # --- Check blowup condition ---
        if cumulative_pnl < trailing_floor:
            results.append({
                "Start_Date": df.loc[start_idx, 'Date'],
                "Rows_to_+Target": None,
                "Max_Drawdown": peak_pnl - cumulative_pnl,
                "Average_Contracts": sum(contract_history) / len(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "Minimum_Contracts": min(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "Maximum_Contracts": max(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "End_Date": df.loc[i, 'Date'],
                "Blown": True
            })
            blown = True
            break

        # --- Check profit target ---
        if cumulative_pnl >= TARGET:
            results.append({
                "Start_Date": df.loc[start_idx, 'Date'],
                "Rows_to_+Target": days,
                "Max_Drawdown": abs(min_cumulative_pnl),
                "Average_Contracts": sum(contract_history) / len(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "Minimum_Contracts": min(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "Maximum_Contracts": max(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "End_Date": df.loc[i, 'Date'],
                "Blown": False
            })
            reached = True
            break

    # --- If we reach the end without hitting either condition ---
    if not reached and not blown:
        results.append({
            "Start_Date": df.loc[start_idx, 'Date'],
            "Rows_to_+Target": None,
            "Max_Drawdown": abs(min_cumulative_pnl),
            "Average_Contracts": sum(contract_history) / len(contract_history) if USE_DYNAMIC_LOT else SIZE,
            "Minimum_Contracts": min(contract_history) if USE_DYNAMIC_LOT else SIZE,
            "Maximum_Contracts": max(contract_history) if USE_DYNAMIC_LOT else SIZE,
            "End_Date": None,
            "Blown": False
        })


# --- Display results ---
results_df = pd.DataFrame(results)
print(results_df)

# --- Stats ---
valid = results_df.dropna(subset=["Rows_to_+Target"])
if not valid.empty:
    print("\n====== SUMMARY STATS ======")
    print("Target:", TARGET)
    print("Max drawdown allowed:", MAX_DD)
    print("Size multiplier:", SIZE)
    print("Dynamic lot enabled:", USE_DYNAMIC_LOT)
    print()
    print("Min days:", valid["Rows_to_+Target"].min())
    print("Max days:", valid["Rows_to_+Target"].max())
    print("Average days:", round(valid["Rows_to_+Target"].mean(), 2))
    print("Median days:", valid["Rows_to_+Target"].median())
    print("Std dev days:", round(valid["Rows_to_+Target"].std(), 2))
    print("Mode days:", valid["Rows_to_+Target"].mode().values)
    print("Count of valid runs:", len(valid))

# --- Probability metrics ---
total_runs = len(results_df)
blowups = len(results_df[results_df["Blown"] == True])
successful = len(valid)

print("\n====== PROBABILITY METRICS ======")
print(f"Total runs: {total_runs}")
print(f"Successful runs: {successful} ({successful / total_runs * 100:.2f}%)")
print(f"Blowups: {blowups} ({blowups / total_runs * 100:.2f}%)")
print(f"Survival probability: {(1 - blowups / total_runs) * 100:.2f}%")

# --- Summary Sheet ---
summary_data = {
    "Metric": [
        "Min days", "Max days", "Average days", "Median days", "Std dev days", "Mode days",
        "Count of valid runs", "Total runs", "Successful runs (%)",
        "Blowups (%)", "Survival probability (%)", "", "Dynamic lot enabled",
        "Position size multiplier", "Target", "Max Drawdown Limit"
    ],
    "Value": [
        valid["Rows_to_+Target"].min() if not valid.empty else None,
        valid["Rows_to_+Target"].max() if not valid.empty else None,
        round(valid["Rows_to_+Target"].mean(), 2) if not valid.empty else None,
        valid["Rows_to_+Target"].median() if not valid.empty else None,
        round(valid["Rows_to_+Target"].std(), 2) if not valid.empty else None,
        valid["Rows_to_+Target"].mode().values[0] if not valid.empty else None,
        len(valid),
        len(results_df),
        f"{len(valid) / len(results_df) * 100:.2f}%" if total_runs > 0 else None,
        f"{blowups / len(results_df) * 100:.2f}%" if total_runs > 0 else None,
        f"{(1 - blowups / len(results_df)) * 100:.2f}%" if total_runs > 0 else None,
        "",
        USE_DYNAMIC_LOT,
        SIZE,
        TARGET,
        MAX_DD
    ]
}
summary_df = pd.DataFrame(summary_data)

# --- Histogram data ---
if not valid.empty:
    hist_data = (
        valid["Rows_to_+Target"]
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"index": "Days", "Rows_to_+Target": "Took_days"})
    )
else:
    hist_data = pd.DataFrame(columns=["Days", "Took_days"])

# --- Save all to Excel ---
folder = "../1500 count/Runs_reports_dynamic" if USE_DYNAMIC_LOT else "../1500 count/Runs_reports_static"
filename = f"dynamic_pnl_growth_report_{TARGET}_{MAX_DD}_{SIZE}_{CONTRACT_STEP}.xlsx" if USE_DYNAMIC_LOT \
    else f"static_pnl_growth_report_{TARGET}_{MAX_DD}_{SIZE}.xlsx"

with pd.ExcelWriter(f"{folder}/{filename}") as writer:
    results_df.to_excel(writer, sheet_name="All Runs", index=False)
    summary_df.to_excel(writer, sheet_name="Summary Stats", index=False)
    hist_data.to_excel(writer, sheet_name="Histogram", index=False)
# --- Optional save of detailed contract log ---

if SAVE_CONTRACT_LOG:
    details_df = pd.DataFrame(detailed_log)
    details_path = f"../1500 count/Logs/dynamic_contracts_log_{TARGET}_{MAX_DD}_{SIZE}_{CONTRACT_STEP}.csv" if USE_DYNAMIC_LOT \
        else f"../1500 count/Logs/static_contracts_log_{TARGET}_{MAX_DD}_{SIZE}.csv"
    details_df.to_csv(details_path, index=False, sep="\t")
    print(f"\n📄 Detailed contract log saved to: {details_path}")


print(f"\n✅ Excel report created: {filename}")
print("   Sheets: [All Runs, Summary Stats, Histogram]")
