import pandas as pd
import matplotlib.pyplot as plt

input_file_path = "daily_results_mae_eq_peak_low.csv"  # input file from MAE/CSV_MT5_trade_stats_to_daily_MAE.py
df = pd.read_csv(input_file_path, sep="\t")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# cumulative equity
df["Equity"] = df["PNL"].cumsum()

# MAE / MFE dots
df["Equity_MAE_dot"] = df["Equity"] + df["MAE"]
df["Equity_MFE_dot"] = df["Equity"] + df["MFE"]

plt.figure(figsize=(11, 6))

# equity curve
plt.plot(df["Date"], df["Equity"], label="Equity (closed)", linewidth=2)

# MAE / MFE dots
# plt.scatter(df["Date"], df["Equity_MAE_dot"], label="MAE (daily worst)", marker="o")
# plt.scatter(df["Date"], df["Equity_MFE_dot"], label="MFE (daily best)", marker="o")

# Equity peak/low curves
plt.plot(df["Date"], df["Equity_Peak"], label="Equity_Peak", linewidth=1)
plt.plot(df["Date"], df["Equity_Low"], label="Equity_Low", linewidth=1)

plt.axhline(0, linewidth=0.8)
plt.title("Equity Curve with Daily MAE / MFE Dots")
plt.xlabel("Date")
plt.ylabel("PNL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
