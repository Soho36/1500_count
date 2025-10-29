import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load and preprocess data ===
# Read the CSV (adjust separator if needed)
df_raw = pd.read_csv('compare.csv', sep='\t')


START_DATE = "2020-04-01"          # set to None to disable filtering "YYYY-MM-DD"
END_DATE = "2025-05-01"             # set to None to disable filtering "YYYY-MM-DD"

# Rename columns clearly
df_raw.columns = ['Date_1', 'PL_1', 'Date_2', 'PL_2']

# Convert dates and numeric data
df_raw['Date_1'] = pd.to_datetime(df_raw['Date_1'], dayfirst=True, errors='coerce')
df_raw['Date_2'] = pd.to_datetime(df_raw['Date_2'], dayfirst=True, errors='coerce')

# Clean numeric values
df_raw['PL_1'] = pd.to_numeric(df_raw['PL_1'].astype(str).str.replace(',', '.'), errors='coerce')
df_raw['PL_2'] = pd.to_numeric(df_raw['PL_2'].astype(str).str.replace(',', '.'), errors='coerce')

# Create two separate DataFrames
df1 = df_raw[['Date_1', 'PL_1']].dropna().rename(columns={'Date_1': 'Date'})
df2 = df_raw[['Date_2', 'PL_2']].dropna().rename(columns={'Date_2': 'Date'})

# === Merge by Date ===
df = pd.merge(df1, df2, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
print(f"Merged {len(df)} rows after aligning dates.")

# === Filter by date range if specified ===
if START_DATE:
    df = df[df['Date'] >= pd.to_datetime(START_DATE)]
if END_DATE:
    df = df[df['Date'] <= pd.to_datetime(END_DATE)]
print(f"Filtered to {len(df)} rows between {START_DATE} and {END_DATE}.")

# Continue with the rest of the script (Combined, Equity, etc.)
df['Combined'] = df['PL_1'] + df['PL_2']
df['Equity_1'] = df['PL_1'].cumsum()
df['Equity_2'] = df['PL_2'].cumsum()
df['Equity_Combined'] = df['Combined'].cumsum()


# === Basic stats ===
corr = df['PL_1'].corr(df['PL_2'])
print(f"\nCorrelation between strategies: {corr:.3f}")


# === Sharpe ratio calculation ===
def sharpe(series):
    return series.mean() / series.std() * np.sqrt(252)  # assuming daily data


print("\nSharpe Ratios:")
print(f"Strategy 1: {sharpe(df['PL_1']):.2f}")
print(f"Strategy 2: {sharpe(df['PL_2']):.2f}")
print(f"Combined:   {sharpe(df['Combined']):.2f}")


# === Drawdown calculation ===
def max_drawdown(equity):
    running_max = equity.cummax()
    dd = equity - running_max
    return dd.min()


print("\nMax Drawdowns:")
print(f"Strategy 1: {max_drawdown(df['Equity_1']):.2f}")
print(f"Strategy 2: {max_drawdown(df['Equity_2']):.2f}")
print(f"Combined:   {max_drawdown(df['Equity_Combined']):.2f}")

# === Plot equity curves ===
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Equity_1'], label='Strategy 1', linewidth=1.5)
plt.plot(df['Date'], df['Equity_2'], label='Strategy 2', linewidth=1.5)
plt.plot(df['Date'], df['Equity_Combined'], label='Combined', linewidth=2, color='black')
plt.title('Equity Curves Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Optional: Rolling correlation ===
df['RollingCorr'] = df['PL_1'].rolling(30).corr(df['PL_2'])
plt.figure(figsize=(10,4))
plt.plot(df['Date'], df['RollingCorr'])
plt.title('30-day Rolling Correlation')
plt.grid(True)
plt.show()
