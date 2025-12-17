import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("daily_results.csv", sep="\t")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# cumulative equity
df["Equity"] = df["PNL"].cumsum()

# MAE / MFE dots (your definition)
df["Equity_MAE_dot"] = df["Equity"] + df["MAE"]
df["Equity_MFE_dot"] = df["Equity"] + df["MFE"]

plt.figure(figsize=(11, 6))

# equity curve
plt.plot(df["Date"], df["Equity"], label="Equity (closed)", linewidth=2)

# MAE / MFE dots
plt.scatter(df["Date"], df["Equity_MAE_dot"], label="MAE (daily worst)", marker="o")
plt.scatter(df["Date"], df["Equity_MFE_dot"], label="MFE (daily best)", marker="o")

plt.axhline(0, linewidth=0.8)
plt.title("Equity Curve with Daily MAE / MFE Dots")
plt.xlabel("Date")
plt.ylabel("PNL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
