import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# === CONFIG ===
MAX_DD = 1500               # maximum drawdown allowed before "blowup"
TARGET = 1500               # profit target per run
SIZE = 1                    # static lot size (if not using dynamic)
COST_PER_MONTH = 40         # cost per month per run

input_file = "CSVS/all_times_14_flat.csv"
# input_file = "CSVS/premarket_only.csv"
# input_file = "CSVS/top_times_only.csv"

# --- Run scheduling mode ---

RUN_MODE = "OVERLAPPING"      # New runs start every day (overlapping)
# RUN_MODE = "SEQUENTIAL"       # New run starts only after previous run ends
# RUN_MODE = "MONTHLY"            # New runs start at beginning of each month

RUNS_PER_MONTH = 2  # how many new runs to start every month (if RUN_MODE = "MONTHLY")

# --- Drawdown options ---
USE_TRAILING_DD = True      # 🔁 switch: True = trailing DD, False = static DD

# --- Dynamic lot options ---
USE_DYNAMIC_LOT = False     # 🔄 switch: True = dynamic lot, False = static
CONTRACT_STEP = 1000         # add/remove 1 contract per $500 gain/loss

# --- Logging options ---
SAVE_CONTRACT_LOG = True    # save detailed per-day info for first N runs
MAX_RUNS_TO_LOG = 1500       # limit detailed log to first N runs

# --- Optional date filter ---

# START_DATE = "2025-09-12"          # set to None to disable filtering "YYYY-MM-DD"
START_DATE = None
END_DATE = None
# END_DATE = "2023-07-29"             # set to None to disable filtering "YYYY-MM-DD"

SHOW_PLOTS = False  # set to True to display plots interactively

dataframe = pd.read_csv(input_file, sep="\t")
input_filename = (os.path.basename(input_file)).replace(".csv", "")
print(f"📊 Loaded data from: {input_file}")

# Output display settings
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 7)

# --- Clean numeric formatting ---
for col in ["P/L", "Net", "Hi", "Low", "Open", "Close"]:
    if col in dataframe.columns:
        dataframe[col] = (
            dataframe[col]
            .astype(str)
            .str.replace(' ', '', regex=False)
            .str.replace(',', '.', regex=False)
            .astype(float)
        )

# --- Parse dates ---
dataframe["Date"] = pd.to_datetime(dataframe["Date"], format="%d.%m.%Y")

# --- Create synthetic "P/L (Net)" column for compatibility ---
# We'll treat daily PnL as the change in Net from previous day
dataframe["P/L (Net)"] = dataframe["Net"].diff().fillna(dataframe["P/L"].iloc[0] if "P/L" in dataframe.columns else 0)

# --- Optional date filter ---
if START_DATE or END_DATE:
    if START_DATE:
        dataframe = dataframe[dataframe["Date"] >= pd.to_datetime(START_DATE)]
    if END_DATE:
        dataframe = dataframe[dataframe["Date"] <= pd.to_datetime(END_DATE)]
    dataframe = dataframe.sort_values("Date").reset_index(drop=True)
    print(f"📅 Data filtered from {START_DATE or 'beginning'} to {END_DATE or 'end'}")
    # print(f"Remaining rows: {len(dataframe)}")

# ==============


results = []
detailed_log = []


def detailed_log_helper(det_log, df, st_idx, ix, cts, pnl_today, cumul_pnl, pk_pnl, trail_floor):
    """Helper to append a single day's run data to the detailed log."""
    det_log.append({
        "Run_Start": df.loc[st_idx, 'Date'],
        "Date": df.loc[ix, 'Date'],
        "Contracts": cts,
        "Cumulative_PnL_Today": round(pnl_today, 2),
        "Cumulative_PnL": round(cumul_pnl, 2),
        "Peak_PnL": round(pk_pnl, 2),
        "Trailing_Floor": round(trail_floor, 2)
    })


# --- Prepare list of start indices depending on mode ---
if RUN_MODE == "MONTHLY":
    dataframe["YearMonth"] = pd.to_datetime(dataframe["Date"]).dt.to_period("M")
    start_indices = (
        dataframe.groupby("YearMonth")
        .head(RUNS_PER_MONTH)
        .index
        .tolist()
    )
else:
    start_indices = list(range(len(dataframe)))

# --- Loop through every possible starting date ---
start_idx_pointer = 0

# --- Main simulation loop ---
while start_idx_pointer < len(start_indices):
    start_idx = start_indices[start_idx_pointer]
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

    # --- iterate through data until run ends or dataset ends ---
    for i in range(start_idx, len(dataframe)):
        # Record contract size
        contract_history.append(contracts)

        # --- Apply today's PnL ---
        cumulative_pnl_today = dataframe.loc[i, 'P/L (Net)'] * contracts
        projected_pnl = cumulative_pnl + cumulative_pnl_today
        days += 1

        # --- Check if today overshoots the target ---
        if projected_pnl >= TARGET:
            # take only the remaining amount needed to reach target
            cumulative_pnl_today = TARGET - cumulative_pnl
            cumulative_pnl = TARGET
            min_cumulative_pnl = min(min_cumulative_pnl, cumulative_pnl)

            # log the truncated last step
            if SAVE_CONTRACT_LOG and start_idx < MAX_RUNS_TO_LOG:
                detailed_log_helper(
                    detailed_log, dataframe, start_idx, i,
                    contracts, cumulative_pnl_today, cumulative_pnl,
                    peak_pnl, trailing_floor
                )

            # record the completed run
            results.append({
                "Start_Date": dataframe.loc[start_idx, 'Date'],
                "Rows_to_+Target": days,
                "Rows_to_blown": None,
                "Max_Drawdown": abs(min_cumulative_pnl),
                "Average_Contracts": sum(contract_history) / len(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "Minimum_Contracts": min(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "Maximum_Contracts": max(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "End_Date": dataframe.loc[i, 'Date'],
                "Blown": False
            })
            reached = True
            break

        # --- otherwise continue normally ---
        cumulative_pnl = projected_pnl
        min_cumulative_pnl = min(min_cumulative_pnl, cumulative_pnl)

        # --- Update contract size dynamically (only if enabled) ---
        if USE_DYNAMIC_LOT:
            contracts = max(1, 1 + int(cumulative_pnl // CONTRACT_STEP))

        # --- Update DD logic ---
        if USE_TRAILING_DD:
            if "Hi" in dataframe.columns and "Net" in dataframe.columns:
                intraday_peak = cumulative_pnl + (dataframe.loc[i, "Hi"] - dataframe.loc[i, "Close"])
                peak_pnl = max(peak_pnl, intraday_peak)
            else:
                peak_pnl = max(peak_pnl, cumulative_pnl)

            trailing_floor = peak_pnl - MAX_DD
            dd_limit = trailing_floor
            dd_breached = cumulative_pnl < dd_limit
        else:
            dd_limit = -MAX_DD
            dd_breached = cumulative_pnl <= dd_limit

        # --- save per-day details ---
        if SAVE_CONTRACT_LOG and start_idx < MAX_RUNS_TO_LOG:
            detailed_log_helper(
                detailed_log, dataframe, start_idx, i,
                contracts, cumulative_pnl_today, cumulative_pnl,
                peak_pnl, trailing_floor
            )

        # --- Check blowup condition ---
        if dd_breached:
            # Cut today’s loss to land exactly on the drawdown limit
            adjustment = dd_limit - cumulative_pnl
            cumulative_pnl_today += adjustment
            cumulative_pnl = dd_limit  # precisely hit DD threshold

            # Log the truncated last step
            if SAVE_CONTRACT_LOG and start_idx < MAX_RUNS_TO_LOG:
                detailed_log_helper(
                    detailed_log, dataframe, start_idx, i,
                    contracts, cumulative_pnl_today, cumulative_pnl,
                    peak_pnl, trailing_floor
                )

            results.append({
                "Start_Date": dataframe.loc[start_idx, 'Date'],
                "Rows_to_+Target": None,
                "Rows_to_blown": days,
                "Max_Drawdown": peak_pnl - cumulative_pnl if USE_TRAILING_DD else abs(min_cumulative_pnl),
                "Average_Contracts": sum(contract_history) / len(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "Minimum_Contracts": min(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "Maximum_Contracts": max(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "End_Date": dataframe.loc[i, 'Date'],
                "Blown": True
            })
            blown = True
            break

    # --- If we reach the end without hitting either condition ---
    if not reached and not blown:
        results.append({
            "Start_Date": dataframe.loc[start_idx, 'Date'],
            "Rows_to_+Target": None,
            "Rows_to_blown": None,
            "Max_Drawdown": abs(min_cumulative_pnl),
            "Average_Contracts": sum(contract_history) / len(contract_history) if USE_DYNAMIC_LOT else SIZE,
            "Minimum_Contracts": min(contract_history) if USE_DYNAMIC_LOT else SIZE,
            "Maximum_Contracts": max(contract_history) if USE_DYNAMIC_LOT else SIZE,
            "End_Date": None,
            "Blown": False
        })

    # --- Advance start index depending on mode ---
    if RUN_MODE == "OVERLAPPING":
        start_idx_pointer += 1

    elif RUN_MODE == "SEQUENTIAL":
        if reached or blown:
            start_idx_pointer += 1  # move to next start index after current run
            if i + 1 < len(dataframe):
                # skip future starts until after current run end
                next_starts = [idx for idx in start_indices if idx > i]
                if next_starts:
                    start_idx_pointer = start_indices.index(next_starts[0])
                else:
                    break
        else:
            start_idx_pointer += 1

    elif RUN_MODE == "MONTHLY":
        start_idx_pointer += 1


# Monthly subtotals of blown runs

results_df = pd.DataFrame(results)
results_df["Start_Date"] = pd.to_datetime(results_df["Start_Date"])

# --- Add a 'Completed' column ---
# A run is 'completed' if it either reached the target or was blown
results_df["Completed"] = results_df["Rows_to_+Target"].notna() | results_df["Rows_to_blown"].notna()
# --- Add YearMonth column ---
results_df["YearMonth"] = results_df["Start_Date"].dt.to_period("M").astype(str)

# Add a YearMonth column and group by it
monthly_stats = (
    results_df.groupby("YearMonth")["Blown"]
    .apply(lambda x: x.sum())  # counts how many True values
    .reset_index(name="Blown_Runs")
)

# Count completed runs per month (blown + successful)
completed_counts = (
    results_df.groupby("YearMonth")["Completed"]
    .apply(lambda x: x.sum())
    .reset_index(name="Completed_Runs")
)
# Merge the two counts into monthly_stats
monthly_stats = monthly_stats.merge(completed_counts, on="YearMonth", how="left")

# Calculate total started runs
monthly_stats["Total_Runs"] = results_df.groupby("YearMonth")["Blown"].count().values

# Calculate percentage of blown out of completed runs
monthly_stats["Blown_%"] = (
    monthly_stats["Blown_Runs"] / monthly_stats["Completed_Runs"].replace(0, np.nan) * 100).round(2)

# Calculate successful runs correctly (completed but not blown)
monthly_stats["Successful_Runs"] = monthly_stats["Completed_Runs"] - monthly_stats["Blown_Runs"]


# --- Compute DD% for each run ---
results_df["DD_%"] = (results_df["Max_Drawdown"] / MAX_DD) * 100
results_df["Days_per_run"] = results_df.apply(lambda row: row["Rows_to_+Target"] if pd.notna(row["Rows_to_+Target"]) else row["Rows_to_blown"], axis=1)

# Make sure Days_per_run is numeric (in days)
results_df["Days_per_run"] = pd.to_numeric(results_df["Days_per_run"], errors="coerce")

# Compute how many 30-day periods the run lasted (ceil division)
results_df["Months_per_run"] = np.ceil(results_df["Days_per_run"] / 30).astype("Int64")

# Calculate total cost
results_df["Run_Cost"] = results_df["Months_per_run"] * COST_PER_MONTH

# Extract year from Start_Date
results_df["Year"] = results_df["Start_Date"].dt.year

# Aggregate total cost per year
yearly_costs = (
    results_df.groupby("Year")["Run_Cost"]
    .sum()
    .reset_index(name="Total_Cost_per_Year")
)

# Merge yearly totals back to main dataframe
results_df = results_df.merge(yearly_costs, on="Year", how="left")

# Yearly cost summary
yearly_summary = (
    results_df.groupby("Year")
    .agg(
        Total_Cost=("Run_Cost", "sum"),
        Total_Runs=("Run_Cost", "count"),
        Avg_Run_Length=("Days_per_run", "mean"),
        Blown_Runs=("Blown", "sum"),
        Successful_Runs=("Blown", lambda x: (x == False).sum()),
    )
    .reset_index()
)

# Add blown percentage
yearly_summary["Blown_%"] = (yearly_summary["Blown_Runs"] / yearly_summary["Total_Runs"] * 100).round(2)


total_row = pd.DataFrame({
    "Year": ["Total"],
    "Total_Cost": [yearly_summary["Total_Cost"].sum()],
    "Total_Runs": [yearly_summary["Total_Runs"].sum()],
    "Avg_Run_Length": [results_df["Days_per_run"].mean()],
    "Blown_Runs": [yearly_summary["Blown_Runs"].sum()],
    "Successful_Runs": [yearly_summary["Successful_Runs"].sum()],
    "Blown_%": [(yearly_summary["Blown_Runs"].sum() / yearly_summary["Total_Runs"].sum() * 100).round(2)]
})

# Combine yearly summary + total row
yearly_summary = pd.concat([yearly_summary, total_row], ignore_index=True)

# --- Plotting_1 ---
# Histogram: Distribution of Max DD Used
plt.figure(figsize=(14, 6))
plt.hist(results_df["DD_%"], bins=20, color="teal", edgecolor="black")
plt.axvline(100, color="red", linestyle="--", label="Max DD limit (100%)")
plt.title(f"Distribution of Maximum Drawdown (% of limit) (Target={TARGET}, MaxDD={MAX_DD}, Size={SIZE})")
plt.xlabel("Max DD (% of limit)")
plt.ylabel("Number of runs")
plt.legend()
plt.tight_layout()


# --- Save histogram ---
# --- Construct save path depending on lot type, overlap mode, and monthly mode ---
if RUN_MODE == "MONTHLY":
    if USE_DYNAMIC_LOT:
        details_path = (
            f"{input_filename}/Runs_reports_dynamic_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
            f"TDD{USE_TRAILING_DD}_monthly_RP{RUNS_PER_MONTH}.xlsx"
        )
    else:
        details_path = (
            f"{input_filename}/Runs_reports_static_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_static_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
            f"TDD{USE_TRAILING_DD}_monthly_RP{RUNS_PER_MONTH}.xlsx"
        )

elif RUN_MODE == "OVERLAPPING":
    if USE_DYNAMIC_LOT:
        details_path = (
            f"{input_filename}/Runs_reports_dynamic_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
            f"TDD{USE_TRAILING_DD}_overlapping.xlsx"
        )
    else:
        details_path = (
            f"{input_filename}/Runs_reports_static_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_static_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
            f"TDD{USE_TRAILING_DD}_overlapping.xlsx"
        )

else:  # Non-overlapping sequential mode
    if USE_DYNAMIC_LOT:
        details_path = (
            f"{input_filename}/Runs_reports_dynamic_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
            f"TDD{USE_TRAILING_DD}_sequential.xlsx"
        )
    else:
        details_path = (
            f"{input_filename}/Runs_reports_static_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_static_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
            f"TDD{USE_TRAILING_DD}_sequential.xlsx"
        )


# Extract directory from details_path and create save path
directory = os.path.dirname(details_path)
filename_without_ext = os.path.splitext(os.path.basename(details_path))[0]
save_path = os.path.join(directory, f"{filename_without_ext}_Distribution_of_Max_DD_Used.png")

# Create directory if it doesn't exist
os.makedirs(directory, exist_ok=True)
plt.savefig(save_path)
print(f"✅ Histogram: Distribution of Max DD Used saved to: {save_path}")

# --- Plotting_2 ---
# Combine Red/Green Blow/Success Bar Chart
colors = results_df["Blown"].map({True: "red", False: "green"})
x_dates = pd.to_datetime(results_df["Start_Date"])

# Invert DD% for blown runs so they point downwards
dd_values = results_df.apply(
    lambda row: -row["DD_%"] if row["Blown"] else row["DD_%"], axis=1
)

# noinspection PyTypeChecker
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Chart 1 — DD%
axs[0].bar(x_dates, dd_values, color=colors)
axs[0].axhline(0, color="black", linestyle="--", linewidth=1)
axs[0].set_ylabel("Max Drawdown (% of limit)")
axs[0].set_title("Red = Blown (down), Green = Success (up)")
axs[0].legend(["Zero line"], loc="upper left")

# Chart 2 — Days per run
axs[1].bar(x_dates, results_df["Days_per_run"], color="dodgerblue", alpha=0.6)
axs[1].set_ylabel("Days per run")
axs[1].set_xlabel("Date")

plt.tight_layout()

# --- Save combined chart ---
if RUN_MODE == "MONTHLY":
    if USE_DYNAMIC_LOT:
        details_path = (
            f"{input_filename}/Runs_reports_dynamic_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
            f"TDD{USE_TRAILING_DD}_monthly_RP{RUNS_PER_MONTH}.xlsx"
        )
    else:
        details_path = (
            f"{input_filename}/Runs_reports_static_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_static_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
            f"TDD{USE_TRAILING_DD}_monthly_RP{RUNS_PER_MONTH}.xlsx"
        )

elif RUN_MODE == "OVERLAPPING":
    if USE_DYNAMIC_LOT:
        details_path = (
            f"{input_filename}/Runs_reports_dynamic_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
            f"TDD{USE_TRAILING_DD}_overlapping.xlsx"
        )
    else:
        details_path = (
            f"{input_filename}/Runs_reports_static_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_static_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
            f"TDD{USE_TRAILING_DD}_overlapping.xlsx"
        )

else:  # Non-overlapping sequential mode
    if USE_DYNAMIC_LOT:
        details_path = (
            f"{input_filename}/Runs_reports_dynamic_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
            f"TDD{USE_TRAILING_DD}_sequential.xlsx"
        )
    else:
        details_path = (
            f"{input_filename}/Runs_reports_static_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_static_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
            f"TDD{USE_TRAILING_DD}_sequential.xlsx"
        )


# Extract directory from details_path and create save path
directory = os.path.dirname(details_path)
filename_without_ext = os.path.splitext(os.path.basename(details_path))[0]
save_path = os.path.join(directory, f'{filename_without_ext}_Combine_Red_Green_Blow_Success_Bar_Chart.png')

# Create directory if it doesn't exist
os.makedirs(directory, exist_ok=True)
plt.savefig(save_path)
print(f"✅ Combine_Red_Green_Blow_Success_Bar_Chart saved to: {save_path}")

# --- Plotting_3 ---
# Year-Monthly Blown Run Statistics chart

monthly_stats.plot(x="YearMonth", y="Blown_Runs", kind="bar", title="Blown Runs per Month", figsize=(14, 8), color="salmon", legend=False)

plt.tight_layout()

# --- Save Year-Monthly Blown Run Statistics chart ---
if RUN_MODE == "MONTHLY":
    if USE_DYNAMIC_LOT:
        details_path = (
            f"{input_filename}/Runs_reports_dynamic_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
            f"TDD{USE_TRAILING_DD}_monthly_RP{RUNS_PER_MONTH}.xlsx"
        )
    else:
        details_path = (
            f"{input_filename}/Runs_reports_static_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_static_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
            f"TDD{USE_TRAILING_DD}_monthly_RP{RUNS_PER_MONTH}.xlsx"
        )

elif RUN_MODE == "OVERLAPPING":
    if USE_DYNAMIC_LOT:
        details_path = (
            f"{input_filename}/Runs_reports_dynamic_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
            f"TDD{USE_TRAILING_DD}_overlapping.xlsx"
        )
    else:
        details_path = (
            f"{input_filename}/Runs_reports_static_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_static_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
            f"TDD{USE_TRAILING_DD}_overlapping.xlsx"
        )

else:
    if USE_DYNAMIC_LOT:
        details_path = (
            f"{input_filename}/Runs_reports_dynamic_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
            f"TDD{USE_TRAILING_DD}_sequential.xlsx"
        )
    else:
        details_path = (
            f"{input_filename}/Runs_reports_static_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_static_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
            f"TDD{USE_TRAILING_DD}_sequential.xlsx"
        )


# Extract directory from details_path and create save path
directory = os.path.dirname(details_path)
filename_without_ext = os.path.splitext(os.path.basename(details_path))[0]
save_path = os.path.join(directory, f'{filename_without_ext}_Year_Monthly_Blown_Run_Chart.png')

# Create directory if it doesn't exist
os.makedirs(directory, exist_ok=True)
plt.savefig(save_path)
print(f"✅ Year_Monthly_Blown_Run_Chart saved to: {save_path}")

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

print("\n=== Monthly Blown Run Statistics ===")
print(monthly_stats.to_string(index=False))

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
if RUN_MODE == "MONTHLY":
    if USE_DYNAMIC_LOT:
        details_path = (
            f"{input_filename}/Runs_reports_dynamic_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
            f"TDD{USE_TRAILING_DD}_monthly_RP{RUNS_PER_MONTH}.xlsx"
        )
    else:
        details_path = (
            f"{input_filename}/Runs_reports_static_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_static_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
            f"TDD{USE_TRAILING_DD}_monthly_RP{RUNS_PER_MONTH}.xlsx"
        )

elif RUN_MODE == "OVERLAPPING":
    if USE_DYNAMIC_LOT:
        details_path = (
            f"{input_filename}/Runs_reports_dynamic_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
            f"TDD{USE_TRAILING_DD}_overlapping.xlsx"
        )
    else:
        details_path = (
            f"{input_filename}/Runs_reports_static_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_static_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
            f"TDD{USE_TRAILING_DD}_overlapping.xlsx"
        )

else:
    if USE_DYNAMIC_LOT:
        details_path = (
            f"{input_filename}/Runs_reports_dynamic_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
            f"TDD{USE_TRAILING_DD}_sequential.xlsx"
        )
    else:
        details_path = (
            f"{input_filename}/Runs_reports_static_lot/"
            f"{START_DATE}_{END_DATE}_{input_filename}_static_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
            f"TDD{USE_TRAILING_DD}_sequential.xlsx"
        )


# Extract directory from details_path and create save path
directory = os.path.dirname(details_path)

# Create directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

with pd.ExcelWriter(f"{details_path}", engine="xlsxwriter") as writer:

    results_df = results_df.sort_values("Start_Date").reset_index(drop=True)
    results_df.to_excel(writer, sheet_name="All Runs", index=False)
    start_row = len(results_df) + 3  # leave a few blank lines
    yearly_summary.to_excel(writer, index=False, sheet_name="All Runs", startrow=start_row)
    summary_df.to_excel(writer, sheet_name="Summary Stats", index=False)
    hist_data.to_excel(writer, sheet_name="Histogram", index=False)
    monthly_stats.to_excel(writer, sheet_name="Monthly Blown Stats", index=False)

    # Set column width for "Summary Stats" sheet
    worksheet_summary = writer.sheets["Summary Stats"]
    worksheet_monthly_blown = writer.sheets["Monthly Blown Stats"]
    worksheet_all_runs = writer.sheets["All Runs"]

    bold_format = writer.book.add_format({"bold": True})  # Define bold format
    worksheet_summary.set_column(0, 0, 25)  # Adjust column A width (Metric column)
    worksheet_summary.set_column(1, 1, 15)  # Adjust column B width (Value column)

    worksheet_summary.set_row(7, None, bold_format)   # Row 1 (index starts at 0)
    worksheet_summary.set_row(21, None, bold_format)  # Row 5
    worksheet_summary.set_row(28, None, bold_format)  # Row 9

    worksheet_monthly_blown.set_column(0, 4, 15)  # Adjust column A width
    # worksheet_monthly_blown.set_column(1, 1, 15)  # Adjust column B width
    # worksheet_monthly_blown.set_column(2, 2, 15)  # Adjust column C width
    # worksheet_monthly_blown.set_column(3, 3, 15)  # Adjust column D width
    # worksheet_monthly_blown.set_column(4, 4, 15)  # Adjust column E width

    worksheet_all_runs.set_column(0, 0, 20)  # Start_Date
    worksheet_all_runs.set_column(7, 7, 20)  # End_Date

# --- Save detailed contract log if enabled ---

if SAVE_CONTRACT_LOG:
    details_df = pd.DataFrame(detailed_log)

    # --- Save detailed contracts log ---
    if RUN_MODE == "MONTHLY":
        if USE_DYNAMIC_LOT:
            details_path = (
                f"{input_filename}/Logs/"
                f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_contracts_log_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
                f"TDD{USE_TRAILING_DD}_monthly_RP{RUNS_PER_MONTH}.csv"
            )
        else:
            details_path = (
                f"{input_filename}/Logs/"
                f"{START_DATE}_{END_DATE}_{input_filename}_static_contracts_log_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
                f"TDD{USE_TRAILING_DD}_monthly_RP{RUNS_PER_MONTH}.csv"
            )

    elif RUN_MODE == "OVERLAPPING":
        if USE_DYNAMIC_LOT:
            details_path = (
                f"{input_filename}/Logs/"
                f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_contracts_log_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
                f"TDD{USE_TRAILING_DD}_overlapping.csv"
            )
        else:
            details_path = (
                f"{input_filename}/Logs/"
                f"{START_DATE}_{END_DATE}_{input_filename}_static_contracts_log_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
                f"TDD{USE_TRAILING_DD}_overlapping.csv"
            )

    else:  # Sequential (non-overlapping) runs
        if USE_DYNAMIC_LOT:
            details_path = (
                f"{input_filename}/Logs/"
                f"{START_DATE}_{END_DATE}_{input_filename}_dynamic_contracts_log_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_STEP{CONTRACT_STEP}_"
                f"TDD{USE_TRAILING_DD}_sequential.csv"
            )
        else:
            details_path = (
                f"{input_filename}/Logs/"
                f"{START_DATE}_{END_DATE}_{input_filename}_static_contracts_log_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}_"
                f"TDD{USE_TRAILING_DD}_sequential.csv"
            )

    os.makedirs(os.path.dirname(details_path), exist_ok=True)
    details_df.to_csv(details_path, index=False, sep="\t")
    print(f"\n📄 Detailed contract log saved to: {details_path}")

report_filename = os.path.basename(details_path)
print(f"\n✅ Excel report created: {report_filename}")
print("   Sheets: [All Runs, Summary Stats, Histogram]")

if SHOW_PLOTS:
    plt.show()
