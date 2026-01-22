import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ==============================
# CONFIG
# ==============================
STARTING_BALANCE = 5000  # Each account starts with $5,000
SAVE_RESULTS = False     # Set to True to save results to Excel

# ==============================
# AUTOMATIC FILE DISCOVERY
# ==============================
# Get all Excel files in the specified directory
folder_path = "../Time_windows_merged_curves/NO_DST_ok_and_good"

# Get list of files with full path
files_with_path = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.endswith(('.xlsx', '.xls'))]

# Get just the filenames for display
file_names_only = [os.path.basename(f) for f in files_with_path]

# ==============================
# LOAD ALL DATA
# ==============================
data_frames = []
loaded_file_names = []

print("\nLoading files...")
for i, (file_path, file_name) in enumerate(zip(files_with_path, file_names_only), 1):
    try:
        # Use the full path to read the file
        df = pd.read_excel(file_path)

        if "Time" not in df.columns or "Balance" not in df.columns:
            print(f"Warning: {file_name} doesn't have required 'Time' or 'Balance' columns")
            continue

        df["Time"] = pd.to_datetime(df["Time"])
        df = df[["Time", "Balance"]].copy()

        # Calculate P&L (Profit & Loss) by subtracting starting balance
        df["P&L"] = df["Balance"] - STARTING_BALANCE

        # Rename column to show it's P&L - use original filename without extension
        file_base = os.path.splitext(file_name)[0]
        df.rename(columns={"P&L": f"P&L_{file_base}"}, inplace=True)

        data_frames.append(df[["Time", f"P&L_{file_base}"]])
        loaded_file_names.append(file_base)
        print(f"✓ Loaded {file_name} as P&L_{file_base}")

    except Exception as e:
        print(f"✗ Error loading {file_name}: {e}")
        continue

if not data_frames:
    print("No files loaded successfully!")
    exit(1)

print(f"\nSuccessfully loaded {len(data_frames)} files")

# ... rest of your code continues ...

# ==============================
# MERGE ALL DATA
# ==============================
print("\nMerging data...")

# Start with first dataframe
merged = data_frames[0].copy()

# Merge all other dataframes
for i in range(1, len(data_frames)):
    merged = pd.merge(
        merged,
        data_frames[i],
        on="Time",
        how="outer"
    )

# Sort by time
merged = merged.sort_values("Time").reset_index(drop=True)

# ==============================
# PROCESS P&L DATA
# ==============================
print("Processing P&L data...")

# Get all P&L columns
pnl_cols = [col for col in merged.columns if col.startswith("P&L_")]

# Forward fill each P&L column (equity curve behavior)
for col in pnl_cols:
    merged[col] = merged[col].ffill()
    # Backfill initial NaNs with first available value
    merged[col].fillna(method="bfill", inplace=True)
    # Fill any remaining NaNs with 0 (starting P&L is 0)
    merged[col].fillna(0, inplace=True)

# Calculate merged P&L (sum of all P&L)
merged["P&L_Merged"] = merged[pnl_cols].sum(axis=1)

# Calculate daily P&L changes
for col in pnl_cols:
    merged[f"Daily_P&L_{col[4:]}"] = merged[col].diff().fillna(0)

# Calculate merged daily P&L
merged["Daily_P&L_Merged"] = merged["P&L_Merged"].diff().fillna(0)

# Calculate P&L as percentage of starting capital
total_starting_capital = STARTING_BALANCE * len(pnl_cols)
merged["P&L_Merged_Pct"] = (merged["P&L_Merged"] / total_starting_capital) * 100

# ==============================
# PROFIT/LOSS STATISTICS
# ==============================
print("\n" + "=" * 60)
print("PROFIT & LOSS ANALYSIS (Starting Balance: $5,000 per account)")
print("=" * 60)

# Final P&L
print(f"\nFINAL PROFIT/LOSS:")
print("-" * 40)
for i, col in enumerate(pnl_cols):
    final_pnl = merged[col].iloc[-1]
    pct_return = (final_pnl / STARTING_BALANCE) * 100
    print(f"  {col}: ${final_pnl:+,.2f} ({pct_return:+.2f}% of $5k capital)")

final_merged_pnl = merged['P&L_Merged'].iloc[-1]
pct_merged_return = (final_merged_pnl / total_starting_capital) * 100
print(f"\n  TOTAL PORTFOLIO P&L: ${final_merged_pnl:+,.2f}")
print(f"  (Combined from {len(pnl_cols)} accounts, ${total_starting_capital:,.0f} total capital)")
print(f"  TOTAL RETURN: {pct_merged_return:+.2f}%")

# Winning/Losing accounts
print(f"\nACCOUNT PERFORMANCE SUMMARY:")
print("-" * 40)
winning_accounts = []
losing_accounts = []
for col in pnl_cols:
    final_pnl = merged[col].iloc[-1]
    if final_pnl > 0:
        winning_accounts.append((col, final_pnl))
    elif final_pnl < 0:
        losing_accounts.append((col, final_pnl))

print(f"  Winning accounts: {len(winning_accounts)}")
if winning_accounts:
    best_account = max(winning_accounts, key=lambda x: x[1])
    print(f"    Best performer: {best_account[0]} (${best_account[1]:+,.2f})")

print(f"  Losing accounts: {len(losing_accounts)}")
if losing_accounts:
    worst_account = min(losing_accounts, key=lambda x: x[1])
    print(f"    Worst performer: {worst_account[0]} (${worst_account[1]:+,.2f})")

print(f"  Break-even accounts: {len(pnl_cols) - len(winning_accounts) - len(losing_accounts)}")

# Risk metrics
print(f"\nRISK METRICS:")
print("-" * 40)

# Maximum drawdown for merged portfolio
merged["Peak_P&L"] = merged["P&L_Merged"].cummax()
merged["Drawdown_$"] = merged["P&L_Merged"] - merged["Peak_P&L"]
merged["Drawdown_Pct"] = (merged["Drawdown_$"] / merged["Peak_P&L"].clip(lower=1)) * 100

max_drawdown = merged["Drawdown_$"].min()
max_drawdown_pct = merged["Drawdown_Pct"].min()
print(f"  Maximum Drawdown: ${max_drawdown:+,.2f} ({max_drawdown_pct:+.2f}% from peak)")

# Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
daily_returns = merged["Daily_P&L_Merged"] / total_starting_capital
sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
print(f"  Sharpe Ratio (annualized): {sharpe_ratio:.3f}")

# Profit Factor
total_profits = merged[merged["Daily_P&L_Merged"] > 0]["Daily_P&L_Merged"].sum()
total_losses = abs(merged[merged["Daily_P&L_Merged"] < 0]["Daily_P&L_Merged"].sum())
profit_factor = total_profits / total_losses if total_losses != 0 else float('inf')
print(f"  Profit Factor: {profit_factor:.2f}")

# Win Rate
winning_days = (merged["Daily_P&L_Merged"] > 0).sum()
total_days = len(merged["Daily_P&L_Merged"])
win_rate = (winning_days / total_days) * 100 if total_days > 0 else 0
print(f"  Win Rate: {win_rate:.1f}% ({winning_days}/{total_days} days)")

# ==============================
# PLOT 1: ALL P&L CURVES
# ==============================
plt.figure(figsize=(16, 8))

# Plot individual P&L curves (with transparency)
colors = plt.cm.tab20(np.linspace(0, 1, len(pnl_cols)))
for col, color in zip(pnl_cols, colors):
    plt.plot(merged["Time"], merged[col], alpha=0.5, linewidth=1, color=color)

# Plot merged P&L (bold)
plt.plot(merged["Time"], merged["P&L_Merged"],
         label=f"Total Portfolio P&L (${final_merged_pnl:+,.0f})",
         color="black", linewidth=3)

# Add horizontal line at 0 (break-even)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.title(f"Profit & Loss Curves - {len(data_frames)} Accounts (${STARTING_BALANCE:,} each)",
          fontsize=14, fontweight='bold')
plt.xlabel("Time", fontsize=12)
plt.ylabel("Profit/Loss ($)", fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
try:
    plt.show()
except KeyboardInterrupt:
    print("\nScript stopped by user.")

# ==============================
# SAVE RESULTS
# ==============================
if SAVE_RESULTS:
    output_file = "portfolio_pnl_analysis.xlsx"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Save P&L data
        merged.to_excel(writer, sheet_name='P&L_Data', index=False)

        # Summary stats
        summary_stats = pd.DataFrame({
            'Metric': [
                'Total Starting Capital',
                'Final Portfolio P&L',
                'Total Return %',
                'Best Account P&L',
                'Worst Account P&L',
                'Winning Accounts',
                'Losing Accounts',
                'Max Drawdown ($)',
                'Max Drawdown %',
                'Sharpe Ratio',
                'Profit Factor',
                'Win Rate %'
            ],
            'Value': [
                f"${total_starting_capital:,.0f}",
                f"${final_merged_pnl:+,.2f}",
                f"{pct_merged_return:+.2f}%",
                f"${best_account[1]:+,.2f}" if winning_accounts else "$0.00",
                f"${worst_account[1]:+,.2f}" if losing_accounts else "$0.00",
                len(winning_accounts),
                len(losing_accounts),
                f"${max_drawdown:+,.2f}",
                f"{max_drawdown_pct:+.2f}%",
                f"{sharpe_ratio:.3f}",
                f"{profit_factor:.2f}",
                f"{win_rate:.1f}%"
            ]
        })
        summary_stats.to_excel(writer, sheet_name='Summary', index=False)

        # Account performance details
        account_perf = []
        for col in pnl_cols:
            final_pnl = merged[col].iloc[-1]
            pct_return = (final_pnl / STARTING_BALANCE) * 100
            max_pnl = merged[col].max()
            min_pnl = merged[col].min()

            account_perf.append({
                'Account': col,
                'Final_P&L': final_pnl,
                'Return_%': pct_return,
                'Max_P&L': max_pnl,
                'Min_P&L': min_pnl,
                'Status': 'Profitable' if final_pnl > 0 else 'Losing' if final_pnl < 0 else 'Break-even'
            })

        pd.DataFrame(account_perf).to_excel(writer, sheet_name='Account_Details', index=False)

    print(f"\nResults saved to: {output_file}")
    print("=" * 60)
    print("PROFIT/LOSS ANALYSIS COMPLETE!")
    print("=" * 60)
else:
    print("\nFinished analysis without saving results.")
    print("=" * 60)
