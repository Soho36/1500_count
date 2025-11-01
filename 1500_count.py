import pandas as pd
import os
import matplotlib.pyplot as plt

# === CONFIG ===
MAX_DD = 1500               # maximum drawdown allowed before "blowup"
TARGET = 1500               # profit target per run
SIZE = 1                    # static lot size (if not using dynamic)


# --- Drawdown options ---
USE_TRAILING_DD = True      # 🔁 switch: True = trailing DD, False = static DD

# --- Dynamic lot options ---
USE_DYNAMIC_LOT = False     # 🔄 switch: True = dynamic lot, False = static
CONTRACT_STEP = 1000         # add/remove 1 contract per $500 gain/loss

# --- Logging options ---
SAVE_CONTRACT_LOG = True    # save detailed per-day info for first N runs
MAX_RUNS_TO_LOG = 1500       # limit detailed log to first N runs

# --- Optional date filter ---

# START_DATE = "2025-05-01"          # set to None to disable filtering "YYYY-MM-DD"
# END_DATE = "2020-02-29"             # set to None to disable filtering "YYYY-MM-DD"
START_DATE = None
END_DATE = None

input_file = "csvs/all_times.csv"
# input_file = "csvs/premarket_only.csv"
# input_file = "csvs/top_times_only.csv"

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
        pnl_today = df.loc[i, 'P/L (Net)'] * (contracts if not USE_DYNAMIC_LOT else contracts)
        projected_pnl = cumulative_pnl + pnl_today
        days += 1

        # --- Check if today overshoots the target ---
        if projected_pnl >= TARGET:
            # take only the remaining amount needed to reach target
            pnl_today = TARGET - cumulative_pnl
            cumulative_pnl = TARGET
            min_cumulative_pnl = min(min_cumulative_pnl, cumulative_pnl)

            # log the truncated last step
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

            # record the completed run
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

        # --- otherwise continue normally ---
        cumulative_pnl = projected_pnl
        min_cumulative_pnl = min(min_cumulative_pnl, cumulative_pnl)

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
                intraday_peak = cumulative_pnl + (df.loc[i, "Hi"] - df.loc[i, "Close"])
                peak_pnl = max(peak_pnl, intraday_peak)
            else:
                peak_pnl = max(peak_pnl, cumulative_pnl)

            trailing_floor = peak_pnl - MAX_DD
            dd_limit = trailing_floor
            dd_breached = cumulative_pnl < dd_limit
        else:
            dd_limit = -MAX_DD
            dd_breached = cumulative_pnl <= dd_limit

        # --- Check blowup condition ---
        if dd_breached:
            # Cut today’s loss to land exactly on the drawdown limit
            overshoot = dd_limit - cumulative_pnl
            pnl_today += overshoot
            cumulative_pnl = dd_limit  # precisely hit DD threshold

            # Log the truncated last step
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


# --- Compute DD% for each run ---
results_df["DD_%"] = (results_df["Max_Drawdown"] / MAX_DD) * 100
results_df["Days_per_run"] = results_df.apply( lambda row: row["Rows_to_+Target"] if pd.notna(row["Rows_to_+Target"]) else row["Rows_to_blown"], axis=1 )
# Histogram: Distribution of Max DD Used
plt.figure(figsize=(14, 6))
plt.hist(results_df["DD_%"], bins=20, color="teal", edgecolor="black")
plt.axvline(100, color="red", linestyle="--", label="Max DD limit (100%)")
plt.title(f"Distribution of Maximum Drawdown (% of limit) (Target={TARGET}, MaxDD={MAX_DD}, Size={SIZE})")
plt.xlabel("Max DD (% of limit)")
plt.ylabel("Number of runs")
plt.legend()
plt.tight_layout()


# Combine Red/Green Blow/Success Bar Chart
colors = results_df["Blown"].map({True: "red", False: "green"})
x_dates = pd.to_datetime(results_df["Start_Date"])

# noinspection PyTypeChecker
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Chart 1 — DD%
axs[0].bar(x_dates, results_df["DD_%"], color=colors)
axs[0].axhline(100, color="black", linestyle="--", label="DD limit (100%)")
axs[0].set_ylabel("Max Drawdown (% of limit)")
axs[0].legend()

# Chart 2 — Days per run
axs[1].bar(x_dates, results_df["Days_per_run"], color="dodgerblue", alpha=0.6)
axs[1].set_ylabel("Days per run")
axs[1].set_xlabel("Date")

plt.tight_layout()


# --- Average maximum DD (as % of limit)
if "Max_Drawdown" in results_df.columns:
    avg_dd = results_df["Max_Drawdown"].mean()
    median_dd = results_df["Max_Drawdown"].median()
    avg_dd_pct = (avg_dd / MAX_DD) * 100
    median_dd_pct = (median_dd / MAX_DD) * 100
else:
    avg_dd = median_dd = avg_dd_pct = median_dd_pct = None


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

# --- Average maximum DD for blown vs non-blown runs ---
non_blown = results_df[results_df["Blown"] == False]
blown = results_df[results_df["Blown"] == True]

avg_dd_all = results_df["Max_Drawdown"].mean()
avg_dd_nonblown = non_blown["Max_Drawdown"].mean() if not non_blown.empty else None
avg_dd_blown = blown["Max_Drawdown"].mean() if not blown.empty else None

avg_dd_all_pct = (avg_dd_all / MAX_DD) * 100
avg_dd_nonblown_pct = (avg_dd_nonblown / MAX_DD) * 100 if avg_dd_nonblown is not None else None
avg_dd_blown_pct = (avg_dd_blown / MAX_DD) * 100 if avg_dd_blown is not None else None

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

    print("\n====== DRAWDOWN UTILIZATION ======")
    print(f"Average Max DD: {avg_dd:.1f}  ({avg_dd_pct:.1f}% of limit)")
    print(f"Median Max DD: {median_dd:.1f}  ({median_dd_pct:.1f}% of limit)")

    print("\n====== DRAWDOWN BREAKDOWN ======")
    print(f"Avg Max DD (all runs): {avg_dd_all:.1f} ({avg_dd_all_pct:.1f}% of limit)")
    print(f"Avg Max DD (non-blown): {avg_dd_nonblown:.1f} ({avg_dd_nonblown_pct:.1f}% of limit)" if avg_dd_nonblown is not None else "Avg Max DD (non-blown): N/A")
    print(f"Avg Max DD (blown): {avg_dd_blown:.1f} ({avg_dd_blown_pct:.1f}% of limit)" if avg_dd_blown is not None else "Avg Max DD (blown): N/A")

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
        "Min days to blowup", "Max days to blowup", "Average days to blowup", "Median days to blowup", "Mode days to blowup",
        "",
        "DRAWDOWN UTILIZATION",
        "Average Max DD",
        "Median Max DD",
        "Average Max DD (%)",
        "Median Max DD (%)",
        "",
        "Average Max DD (all runs)",
        "Average Max DD (non-blown)",
        "Average Max DD (blown)",
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
        mode_blow_days[0] if len(mode_blow_days) > 0 else None,
        "",
        "",
        avg_dd,
        median_dd,
        f"{avg_dd_pct:.1f}%",
        f"{median_dd_pct:.1f}%",
        "",
        avg_dd_all_pct,
        avg_dd_nonblown_pct,
        avg_dd_blown_pct
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
    f"{START_DATE}_{input_filename}_dynamic_pnl_growth_report_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_TDD{USE_TRAILING_DD}.xlsx" if USE_DYNAMIC_LOT \
    else f"{START_DATE}_{input_filename}_static_pnl_growth_report_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_TDD{USE_TRAILING_DD}.xlsx"

os.makedirs(folder, exist_ok=True)
with pd.ExcelWriter(f"{folder}/{filename}", engine="xlsxwriter") as writer:
    results_df = results_df.sort_values("Start_Date").reset_index(drop=True)
    results_df.to_excel(writer, sheet_name="All Runs", index=False)
    summary_df.to_excel(writer, sheet_name="Summary Stats", index=False)
    hist_data.to_excel(writer, sheet_name="Histogram", index=False)

    # Set column width for "Summary Stats" sheet
    worksheet = writer.sheets["Summary Stats"]
    bold_format = writer.book.add_format({"bold": True})  # Define bold format

    worksheet.set_column(0, 0, 25)  # Adjust column A width (Metric column)
    worksheet.set_column(1, 1, 15)  # Adjust column B width (Value column)

    worksheet.set_row(7, None, bold_format)   # Row 1 (index starts at 0)
    worksheet.set_row(21, None, bold_format)  # Row 5
    worksheet.set_row(28, None, bold_format)  # Row 9

if SAVE_CONTRACT_LOG:
    details_df = pd.DataFrame(detailed_log)
    details_path = \
        f"{input_filename}/Logs/{START_DATE}_{input_filename}_dynamic_contracts_log_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_TDD{USE_TRAILING_DD}.csv" if USE_DYNAMIC_LOT \
        else f"{input_filename}/Logs/{START_DATE}_{input_filename}_static_contracts_log_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_TDD{USE_TRAILING_DD}.csv"

    os.makedirs(os.path.dirname(details_path), exist_ok=True)
    details_df.to_csv(details_path, index=False, sep="\t")
    print(f"\n📄 Detailed contract log saved to: {details_path}")


print(f"\n✅ Excel report created: {filename}")
print("   Sheets: [All Runs, Summary Stats, Histogram]")

save_path = f"{folder}/{filename.replace('.xlsx', '_drawdown_utilization.png')}"
print(f"✅ Drawdown utilization chart saved to: {save_path}")
plt.savefig(save_path)
plt.show()
