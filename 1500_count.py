import pandas as pd
import os

# --- Load & clean data ---
# input_file = "csvs/all_times.csv"
input_file = "csvs/all_times_2.csv"
# input_file = "csvs/only_night.csv"
# input_file = "csvs/top_times.csv"

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

# print("✅ Data columns loaded:", list(df.columns))
# print(df.head())

# === CONFIG ===
MAX_DD = 3000               # maximum drawdown allowed before "blowup"
TARGET = 3000               # profit target per run

SIZE = 1                    # static lot size (if not using dynamic)

USE_TRAILING_DD = True      # 🔁 switch: True = trailing DD, False = static DD
USE_DYNAMIC_LOT = False     # 🔄 switch: True = dynamic lot, False = static
CONTRACT_STEP = 1000         # add/remove 1 contract per $500 gain/loss

# --- Logging options ---
SAVE_CONTRACT_LOG = False    # save detailed per-day info for first N runs
MAX_RUNS_TO_LOG = 200       # limit detailed log to first N runs

# --- Optional date filter ---
# START_DATE = "2025-02-01"
START_DATE = None
END_DATE = None

if START_DATE or END_DATE:
    if START_DATE:
        df = df[df["Date"] >= pd.to_datetime(START_DATE)]
    if END_DATE:
        df = df[df["Date"] <= pd.to_datetime(END_DATE)]
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"📅 Data filtered from {START_DATE or 'beginning'} to {END_DATE or 'end'}")
    print(f"Remaining rows: {len(df)}")

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

        # --- Update DD logic (using intraday highs if available) ---
        if USE_TRAILING_DD:
            if "Hi" in df.columns and "Net" in df.columns:
                # Include intraday equity highs for a more realistic trailing DD
                intraday_peak = cumulative_pnl + (df.loc[i, "Hi"] - df.loc[i, "Close"])
                peak_pnl = max(peak_pnl, intraday_peak)
            else:
                # Fallback to normal behavior if columns are missing
                peak_pnl = max(peak_pnl, cumulative_pnl)

            trailing_floor = peak_pnl - MAX_DD
            dd_breached = cumulative_pnl < trailing_floor
        else:
            dd_breached = cumulative_pnl <= -MAX_DD

        # --- Check blowup condition ---
        if dd_breached:
            results.append({
                "Start_Date": df.loc[start_idx, 'Date'],
                "Rows_to_+Target": None,
                "Rows_to_blown": days,
                "Max_Drawdown": peak_pnl - cumulative_pnl if USE_TRAILING_DD else abs(min_cumulative_pnl),
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
                "Rows_to_blown": None,
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
            "Rows_to_blown": None,
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

# --- Blowup stats ---
blown_df = results_df[results_df['Rows_to_blown'].notna()]

if not blown_df.empty:
    min_days_to_blow = blown_df['Rows_to_blown'].min()
    max_days_to_blow = blown_df['Rows_to_blown'].max()
    avg_blow_days = round(blown_df['Rows_to_blown'].mean(), 1)
    median_blow_days = blown_df['Rows_to_blown'].median()
    mode_blow_days = blown_df['Rows_to_blown'].mode().values
else:
    min_days_to_blow = max_days_to_blow = None
    avg_blow_days = median_blow_days = None
    mode_blow_days = []

# --- Stats ---
valid = results_df.dropna(subset=["Rows_to_+Target"])
mode_days = valid["Rows_to_+Target"].mode().values
if not valid.empty:
    print("\n====== SUMMARY STATS ======")
    print("Target:", TARGET)
    print("Max drawdown allowed:", MAX_DD)
    print("Size multiplier:", SIZE)
    print("Dynamic lot enabled:", USE_DYNAMIC_LOT)
    print("Trailing DD enabled:", USE_TRAILING_DD)

    print("\n====== TARGETS ======")
    print("Min days:", valid["Rows_to_+Target"].min())
    print("Max days:", valid["Rows_to_+Target"].max())
    print("Average days:", round(valid["Rows_to_+Target"].mean(), 2))
    print("Median days:", valid["Rows_to_+Target"].median())
    print("Std dev days:", round(valid["Rows_to_+Target"].std(), 2))
    print(f"Mode days: {mode_days[0]:.0f}" if len(mode_days) > 0 else "Mode days: N/A")
    print("Count of valid runs:", len(valid))

    print("\n====== BLOWUPS ======")
    print("Min days to blowup:", min_days_to_blow if min_days_to_blow is not None else "N/A")
    print("Max days to blowup:", max_days_to_blow if max_days_to_blow is not None else "N/A")
    print(f"Avg days to blowup: {avg_blow_days:.0f}" if avg_blow_days is not None else "Avg days to blowup: N/A")
    print(f"Median days to blowup: {median_blow_days:.0f}" if median_blow_days is not None else "Median days to blowup: N/A")
    print(f"Mode days to blowup: {mode_blow_days[0]:.0f}" if len(mode_blow_days) > 0 else "Mode days to blowup: N/A")


# --- Probability metrics ---
total_runs = len(results_df)
blowups = len(results_df[results_df["Blown"] == True])
successful = len(valid)
resolved_runs = successful + blowups

print("\n====== PROBABILITY METRICS ======")
print(f"Total runs (including unfinished): {total_runs}")
print(f"Completed runs: {resolved_runs}")
print(f"Successful runs: {successful} ({successful / resolved_runs * 100:.2f}%)" if resolved_runs > 0 else "Successful runs: N/A")
print(f"Blowups: {blowups} ({blowups / resolved_runs * 100:.2f}%)" if resolved_runs > 0 else "Blowups: N/A")
print(f"Survival probability: {(1 - blowups / resolved_runs) * 100:.2f}%" if resolved_runs > 0 else "Survival probability: N/A")


# --- Summary Sheet ---
summary_data = {
    "Metric": [
        "Dynamic lot",
        "Trailing DD",
        "Position size multiplier",
        "Target",
        "Max Drawdown Limit",
        "",
        "TARGETS STATISTICS",
        "Min days", "Max days", "Average days", "Median days", "Std dev days", "Mode days",
        "Total runs",
        "Resolved runs",
        "Successful runs",
        "Blowups",
        "Successful runs (%)", "Blowups (%)",
        "",
        "BLOWUPS STATISTICS",
        "Min days to blowup", "Max days to blowup", "Average days to blowup", "Median days to blowup", "Mode days to blowup"
    ],
    "Value": [
        USE_DYNAMIC_LOT,
        USE_TRAILING_DD,
        SIZE,
        TARGET,
        MAX_DD,
        "",
        "",
        valid["Rows_to_+Target"].min() if not valid.empty else None,
        valid["Rows_to_+Target"].max() if not valid.empty else None,
        round(valid["Rows_to_+Target"].mean(), 2) if not valid.empty else None,
        valid["Rows_to_+Target"].median() if not valid.empty else None,
        round(valid["Rows_to_+Target"].std(), 2) if not valid.empty else None,
        valid["Rows_to_+Target"].mode().values[0] if not valid.empty else None,
        total_runs,
        resolved_runs,
        len(valid),
        blowups,
        f"{successful / resolved_runs * 100:.1f}%" if resolved_runs > 0 else None,
        f"{blowups / resolved_runs * 100:.1f}%" if resolved_runs > 0 else None,
        "",
        "",
        min_days_to_blow,
        max_days_to_blow,
        avg_blow_days,
        median_blow_days,
        mode_blow_days[0] if len(mode_blow_days) > 0 else None
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
folder = f"{input_filename}/Runs_reports_dynamic_lot" if USE_DYNAMIC_LOT else f"{input_filename}/Runs_reports_static_lot"
filename = \
    f"{input_filename}_dynamic_pnl_growth_report_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_TDD{USE_TRAILING_DD}.xlsx" if USE_DYNAMIC_LOT \
    else f"{input_filename}_static_pnl_growth_report_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_TDD{USE_TRAILING_DD}.xlsx"

os.makedirs(folder, exist_ok=True)  # Ensure folder exists
with pd.ExcelWriter(f"{folder}/{filename}", engine="xlsxwriter") as writer:
    results_df.to_excel(writer, sheet_name="All Runs", index=False)
    summary_df.to_excel(writer, sheet_name="Summary Stats", index=False)
    hist_data.to_excel(writer, sheet_name="Histogram", index=False)

    # Set column width for "Summary Stats" sheet
    worksheet = writer.sheets["Summary Stats"]
    bold_format = writer.book.add_format({"bold": True})  # Define bold format

    worksheet.set_column(0, 0, 25)  # Adjust column A width (Metric column)
    worksheet.set_column(1, 1, 15)  # Adjust column B width (Value column)

    worksheet.set_row(7, None, bold_format)   # Row 1 (index starts at 0)
    worksheet.set_row(22, None, bold_format)  # Row 5

if SAVE_CONTRACT_LOG:
    details_df = pd.DataFrame(detailed_log)
    details_path = \
        f"{input_filename}/Logs/{input_filename}_dynamic_contracts_log_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_TDD{USE_TRAILING_DD}.csv" if USE_DYNAMIC_LOT \
        else f"{input_filename}/Logs/{input_filename}_static_contracts_log_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_TDD{USE_TRAILING_DD}.csv"

    os.makedirs(os.path.dirname(details_path), exist_ok=True)
    details_df.to_csv(details_path, index=False, sep="\t")
    print(f"\n📄 Detailed contract log saved to: {details_path}")


print(f"\n✅ Excel report created: {filename}")
print("   Sheets: [All Runs, Summary Stats, Histogram]")
