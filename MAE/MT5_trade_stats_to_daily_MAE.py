import pandas as pd

# ======================
#  LOAD & CLEAN DATA
# ======================
try:
    input_path = "../MAE/trade_stats.csv"  # input file from MT5 strategy tester
    df = pd.read_csv(input_path, sep="\t")

    for col in ["MAE", "MFE", "PNL"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(' ', '', regex=False)
                .str.replace(',', '.', regex=False)
                .astype(float)
            )

    df["Entry_time"] = pd.to_datetime(df["Entry_time"], format="mixed", dayfirst=True)
    df["Exit_time"] = pd.to_datetime(df["Exit_time"], format="mixed", dayfirst=True)

    # Keep only the necessary columns
    df = df[["Entry_time", "Exit_time", "MAE", "MFE", "PNL"]].dropna()
except FileNotFoundError:
    print("Input file not found. Please ensure 'MAE/trade_stats.csv' exists.")
    exit()

# parse time
df["Entry_time"] = pd.to_datetime(df["Entry_time"])
df["Date"] = df["Entry_time"].dt.date

# ======================
results = []

for date, day in df.groupby("Date"):
    day = day.sort_values("Entry_time")

    equity = 0.0
    day_min = 0.0
    day_max = 0.0

    for _, r in day.iterrows():
        # excursions during trade
        day_min = min(day_min, equity + r["MAE"])
        day_max = max(day_max, equity + r["MFE"])

        # close trade
        equity += r["PNL"]

        # equity after close also counts
        day_min = min(day_min, equity)
        day_max = max(day_max, equity)

    results.append({
        "Date": date.strftime("%d.%m.%Y"),
        "PNL": round(equity, 2),
        "MAE": round(day_min, 2),
        "MFE": round(day_max, 2),
    })

# ======================
# SAVE DAILY RESULTS
# ======================

out = pd.DataFrame(results)
output_file = "../MAE/daily_results.csv"
out.to_csv(output_file, index=False, sep="\t", encoding="utf-8")
print(f"Daily results saved to: {output_file}")
