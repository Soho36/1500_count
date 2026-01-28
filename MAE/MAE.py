import pandas as pd
import matplotlib.pyplot as plt

input_file_path = "arch_csvs/daily_results_mae_eq_peak_low_1.csv"  # input file from MAE/CSV_MT5_trade_stats_to_daily_MAE.py
df = pd.read_csv(input_file_path, sep="\t")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# cumulative equity
df["Equity"] = df["PNL"].cumsum()

# ======================
# DRAWDOWN CALCULATIONS
# ======================

# 1) Closed equity drawdown
df["Equity_Peak_Closed"] = df["Equity"].cummax()
df["DD_Closed"] = df["Equity"] - df["Equity_Peak_Closed"]

# 2) Floating (MAE/MFE-based) drawdown
df["DD_Floating"] = df["Equity_Low"] - df["Equity_Peak"]


# MAE / MFE dots
df["Equity_MAE_dot"] = df["Equity"] + df["MAE"]
df["Equity_MFE_dot"] = df["Equity"] + df["MFE"]

#   MAX DD STATISTICS
print("Max closed DD:", df["DD_Closed"].min())
print("Max floating DD:", df["DD_Floating"].min())

sanity_check = df["DD_Floating"].min() <= df["DD_Closed"].min()
if not sanity_check:
    print("NB! Logic error in MAE/MFE handling")


plt.figure(figsize=(11, 6))

# equity curve
plt.plot(df["Date"], df["Equity"], label="Equity (closed)", linewidth=2)

# MAE / MFE dots
plt.scatter(df["Date"], df["Equity_MAE_dot"], label="MAE (daily worst)", marker="o")
plt.scatter(df["Date"], df["Equity_MFE_dot"], label="MFE (daily best)", marker="o")

# noinspection PyTypeChecker
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# ======================
# 1) Equity curves
# ======================
# Equity (closed) curve
axes[0].plot(df["Date"], df["Equity"], linewidth=2, label="Equity")
# Equity peak/low curves
axes[0].plot(df["Date"], df["Equity_Peak"], linewidth=1, label="Equity_Peak")
axes[0].plot(df["Date"], df["Equity_Low"], linewidth=1, label="Equity_Low")
axes[0].set_title("Equity Curve")
axes[0].set_ylabel("PNL")
axes[0].grid(True)
axes[0].legend()


# ======================
# 2) Closed equity DD
# ======================
axes[1].plot(df["Date"], df["DD_Closed"], linewidth=2, label="Closed DD")
axes[1].axhline(0, linewidth=0.8)
axes[1].set_title("Closed Equity Drawdown")
axes[1].set_ylabel("PNL")
axes[1].grid(True)
axes[1].legend()

# ======================
# 3) Floating (prop-style) DD
# ======================
axes[2].plot(df["Date"], df["DD_Floating"], linewidth=2, label="Floating DD")
axes[2].axhline(0, linewidth=0.8)
axes[2].set_title("OLD")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("PNL")
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()
