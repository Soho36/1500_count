import pandas as pd
import os
# =========================================================
# === CONFIGURATION =======================================
# =========================================================

INPUT_FILE = "csvs/all_times.csv"
# INPUT_FILE = "csvs/only_night.csv"
# INPUT_FILE = "csvs/top_times.csv"

SEP = "\t"
input_filename = (os.path.basename(INPUT_FILE)).replace(".csv", "")

MAX_DD = 3000
TARGET = 3000
SIZE = 1
CONTRACT_STEP = 500
USE_DYNAMIC_LOT = False
USE_TRAILING_DD = True
SAVE_CONTRACT_LOG = False       # disable to speed up monthly runs
MAX_RUNS_TO_LOG = 100
EXPORT_MONTHLY_SUMMARY = True

# =========================================================
# === LOAD AND CLEAN DATA ================================
# =========================================================

df = pd.read_csv(INPUT_FILE, sep=SEP)
print(f"📊 Loaded data from: {INPUT_FILE}")

df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")

df["P/L (Net)"] = (
    df["P/L (Net)"]
    .astype(str)
    .str.replace(" ", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# =========================================================
# === CORE SIMULATION FUNCTION ============================
# =========================================================


def run_simulation(df_slice, start_label=None):
    """Run a single simulation over the given dataframe slice."""
    results = []
    detailed_log = []

    for start_idx in range(len(df_slice)):
        cumulative_pnl = 0
        min_cumulative_pnl = 0
        days = 0
        reached = False
        blown = False
        peak_pnl = 0
        # trailing_floor = -MAX_DD
        contracts = SIZE if not USE_DYNAMIC_LOT else 1
        contract_history = []

        for i in range(start_idx, len(df_slice)):
            pnl_today = df_slice.loc[i, "P/L (Net)"] * (contracts if USE_DYNAMIC_LOT else SIZE)
            cumulative_pnl += pnl_today
            min_cumulative_pnl = min(min_cumulative_pnl, cumulative_pnl)
            days += 1
            contract_history.append(contracts)

            # Update contract size dynamically
            if USE_DYNAMIC_LOT:
                contracts = max(1, 1 + int(cumulative_pnl // CONTRACT_STEP))

            # Update trailing DD logic
            if USE_TRAILING_DD:
                peak_pnl = max(peak_pnl, cumulative_pnl)
                trailing_floor = peak_pnl - MAX_DD
                dd_breached = cumulative_pnl < trailing_floor
            else:
                dd_breached = cumulative_pnl <= -MAX_DD

            # --- Blowup condition ---
            if dd_breached:
                results.append({
                    "Start_Date": df_slice.loc[start_idx, "Date"],
                    "Rows_to_+Target": None,
                    "Rows_to_blown": days,
                    "Max_Drawdown": peak_pnl - cumulative_pnl if USE_TRAILING_DD else abs(min_cumulative_pnl),
                    "Average_Contracts": sum(contract_history)/len(contract_history) if USE_DYNAMIC_LOT else SIZE,
                    "End_Date": df_slice.loc[i, "Date"],
                    "Blown": True
                })
                blown = True
                break

            # --- Profit target ---
            if cumulative_pnl >= TARGET:
                results.append({
                    "Start_Date": df_slice.loc[start_idx, "Date"],
                    "Rows_to_+Target": days,
                    "Rows_to_blown": None,
                    "Max_Drawdown": abs(min_cumulative_pnl),
                    "Average_Contracts": sum(contract_history)/len(contract_history) if USE_DYNAMIC_LOT else SIZE,
                    "End_Date": df_slice.loc[i, "Date"],
                    "Blown": False
                })
                reached = True
                break

        # --- Unfinished run ---
        if not reached and not blown:
            results.append({
                "Start_Date": df_slice.loc[start_idx, "Date"],
                "Rows_to_+Target": None,
                "Rows_to_blown": None,
                "Max_Drawdown": abs(min_cumulative_pnl),
                "Average_Contracts": sum(contract_history)/len(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "End_Date": None,
                "Blown": False
            })

    # === Post-analysis ===
    results_df = pd.DataFrame(results)
    blown_df = results_df[results_df["Rows_to_blown"].notna()]
    valid = results_df.dropna(subset=["Rows_to_+Target"])
    total_runs = len(results_df)
    blowups = len(results_df[results_df["Blown"] == True])
    successful = len(valid)
    resolved_runs = successful + blowups

    # --- Blowup stats ---
    if not blown_df.empty:
        min_days_to_blow = blown_df["Rows_to_blown"].min()
        max_days_to_blow = blown_df["Rows_to_blown"].max()
        avg_blow_days = round(blown_df["Rows_to_blown"].mean(), 1)
        median_blow_days = blown_df["Rows_to_blown"].median()
        mode_blow_days = blown_df["Rows_to_blown"].mode().values
    else:
        min_days_to_blow = max_days_to_blow = avg_blow_days = median_blow_days = None
        mode_blow_days = []

    # === Summary ===
    summary = {
        "Start_Month": start_label or df_slice["Date"].iloc[0].strftime("%Y-%m"),
        "TARGETS STATISTICS": "",
        "Min_days": valid["Rows_to_+Target"].min() if not valid.empty else None,
        "Max_days": valid["Rows_to_+Target"].max() if not valid.empty else None,
        "Avg_days": round(valid["Rows_to_+Target"].mean(), 2) if not valid.empty else None,
        "Median_days": valid["Rows_to_+Target"].median() if not valid.empty else None,
        "Std_dev_days": round(valid["Rows_to_+Target"].std(), 2) if not valid.empty else None,
        "Mode_days": valid["Rows_to_+Target"].mode().iloc[0] if not valid.empty and not valid["Rows_to_+Target"].mode().empty else None,
        "": "",
        "Total_runs": total_runs,
        "Resolved_runs": resolved_runs,
        "Successful_runs": successful,
        "Blowups": blowups,
        "Successful_%": round(successful / resolved_runs * 100, 2) if resolved_runs > 0 else None,
        "Blowups_%": round(blowups / resolved_runs * 100, 2) if resolved_runs > 0 else None,
        "BLOWUPS STATISTICS": "",
        "Min_days_to_blow": min_days_to_blow,
        "Max_days_to_blow": max_days_to_blow,
        "Avg_days_to_blow": avg_blow_days,
        "Median_days_to_blow": median_blow_days,
        "Mode_days_to_blow": mode_blow_days[0] if len(mode_blow_days) > 0 else None
    }

    return summary


# =========================================================
# === MONTHLY ROLLING SIMULATION ==========================
# =========================================================

monthly_summaries = []

unique_months = sorted(df["Date"].dt.to_period("M").unique())

for month in unique_months:
    start_date = month.to_timestamp()
    df_slice = df[df["Date"] >= start_date].reset_index(drop=True)
    if len(df_slice) < 10:
        continue  # skip too-short slices
    print(f"▶ Running simulation starting from {month}")
    summary = run_simulation(df_slice, start_label=str(month))
    monthly_summaries.append(summary)

# =========================================================
# === SAVE TO EXCEL =======================================
# =========================================================

summary_df = pd.DataFrame(monthly_summaries)

# Transpose it: months as columns, metrics as rows
pivoted_df = summary_df.set_index("Start_Month").T

folder_name = f"{input_filename}/Rolling_monthly_reports/"
file_name = (
    f"{input_filename}_Rolling_Monthly_Report_TR{TARGET}_DD{MAX_DD}_SZ{SIZE}"
    f"_DYN_{USE_DYNAMIC_LOT}_TDD{USE_TRAILING_DD}.xlsx"
)
with pd.ExcelWriter(f"{folder_name}/ {file_name}", engine="xlsxwriter") as writer:
    pivoted_df.to_excel(writer,sheet_name="Rolling_months", index=True)

    # Set column width for "Summary Stats" sheet
    worksheet = writer.sheets["Rolling_months"]
    worksheet.set_column(0, 0, 25)  # Set width of first column
    worksheet.set_row(4, None, writer.book.add_format({"bold": True, "font_color": "orange"}))
    worksheet.set_row(13, None, writer.book.add_format({"font_color": "green"}))
    worksheet.set_row(14, None, writer.book.add_format({"font_color": "red"}))

    print(f"\n✅ Monthly rolling simulation completed.")
    print(f"📄 Saved summary report: {file_name}")
