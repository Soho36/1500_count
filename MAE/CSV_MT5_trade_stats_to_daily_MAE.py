import pandas as pd

# ======================
#  LOAD & CLEAN DATA
# ======================
try:
    input_path = "trade_stats_1_minute.csv"  # input file from MT5 strategy tester
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

    df = df[["Entry_time", "Exit_time", "MAE", "MFE", "PNL"]].dropna()
except FileNotFoundError:
    print("Input file not found. Please ensure 'MAE/trade_stats_1_minute.csv' exists.")
    exit()

df["Date"] = df["Entry_time"].dt.date

# ======================
# DAILY MAE / MFE LOGIC
# ======================
results = []

for date, day in df.groupby("Date"):
    day = day.sort_values("Entry_time")

    equity = 0.0    # reset equity to zero at the start of each day
    day_min = 0.0
    day_max = 0.0

    for _, r in day.iterrows():
        # worst/best floating during trade
        day_min = min(day_min, equity + r["MAE"])
        day_max = max(day_max, equity + r["MFE"])

        # close trade
        equity += r["PNL"]

        # closed equity also matters
        day_min = min(day_min, equity)
        day_max = max(day_max, equity)

    results.append({
        "Date": date,
        "PNL": round(equity, 2),
        "MAE": round(day_min, 2),
        "MFE": round(day_max, 2),
    })

daily = pd.DataFrame(results).sort_values("Date")

# ======================
# PROP-FIRM TRAILING DD
# ======================

equity_close = 0.0      # cumulative closed PNL
equity_peak = 0.0       # highest floating equity ever
worst_dd = 0.0          # worst trailing DD observed

prop_rows = []

for _, r in daily.iterrows():
    # highest floating equity can come from MFE
    equity_peak = max(equity_peak, equity_close + r["MFE"])

    # worst floating equity today
    equity_low = equity_close + r["MAE"]

    # trailing DD (prop logic)
    trailing_dd = equity_low - equity_peak
    worst_dd = min(worst_dd, trailing_dd)

    # close day
    equity_close += r["PNL"]

    prop_rows.append({
        "Date": r["Date"].strftime("%d.%m.%Y"),
        "PNL": r["PNL"],
        "MAE": r["MAE"],
        "MFE": r["MFE"],
        "Equity_Close": round(equity_close, 2),
        "Equity_Peak": round(equity_peak, 2),
        "Equity_Low": round(equity_low, 2),
        "Trailing_DD": round(trailing_dd, 2),
        "Worst_DD_So_Far": round(worst_dd, 2),
    })

prop_df = pd.DataFrame(prop_rows)


# ======================
# SAVE OUTPUTS
# ======================

daily_output_mae = "../MAE/daily_results_mae.csv"
daily_output_mae_equity_peak_low = "../MAE/daily_results_mae_eq_peak_low.csv"

daily.assign(Date=daily["Date"].apply(lambda d: d.strftime("%d.%m.%Y"))) \
     .to_csv(daily_output_mae, index=False, sep="\t", encoding="utf-8")

prop_df.to_csv(daily_output_mae_equity_peak_low, index=False, sep="\t", encoding="utf-8")

print(f"Daily MAE/MFE saved to: {daily_output_mae}")
print(f"Prop trailing DD saved to: {daily_output_mae_equity_peak_low}")
