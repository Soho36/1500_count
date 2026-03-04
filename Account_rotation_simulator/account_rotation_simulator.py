import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====== PARAMETERS ======
NUM_ACCOUNTS = 5
START_BALANCE = 1500
MAX_DD = -1500  # relative to start
CSV_PATH = "C:/Users/Liikurserv/PycharmProjects/1500_count/MAE/RG_h1_intervals_night.csv"

# ====== LOAD DATA ======
df = pd.read_csv(CSV_PATH, sep="\t")
df["Entry_time"] = pd.to_datetime(df["Entry_time"])
df = df.sort_values("Entry_time")

# Extract unique trading days in chronological order
unique_days = df["Entry_time"].dt.date.drop_duplicates().tolist()

# Create cyclic account assignment per trading day
day_to_account = {
    day: i % NUM_ACCOUNTS
    for i, day in enumerate(unique_days)
}

trade_pnls = df["PNL"].values
trade_days = df["Entry_time"].dt.date.values
total_trades = len(trade_pnls)

print("=" * 70)
print("TRADING SIMULATION")
print("=" * 70)
print(f"Loaded {total_trades} trades from {CSV_PATH}")
print(f"Starting with {NUM_ACCOUNTS} accounts of ${START_BALANCE} each")
print(f"Total starting capital: ${NUM_ACCOUNTS * START_BALANCE}")
print(f"Max drawdown allowed per account: ${MAX_DD} from start")
print("=" * 70)


def simulate(mode, start_index=2315):
    balances = np.full(NUM_ACCOUNTS, START_BALANCE)
    alive = np.ones(NUM_ACCOUNTS, dtype=bool)
    peak = np.full(NUM_ACCOUNTS, START_BALANCE)
    equity_history = [[START_BALANCE] for _ in range(NUM_ACCOUNTS)]

    first_blow_trade = None
    trades_executed = 0
    trades_skipped = 0

    for i, pnl in enumerate(trade_pnls[start_index:]):

        if not alive.any():
            break

        # === ACCOUNT SELECTION ===
        if mode == "parallel":
            account_indices = np.where(alive)[0]

        elif mode == "sequential_trade":
            idx = i % NUM_ACCOUNTS
            if alive[idx]:
                account_indices = [idx]
            else:
                account_indices = []
                trades_skipped += 1

        elif mode == "one_per_day":
            day_index = day_to_account[trade_days[i]]
            if alive[day_index]:
                account_indices = [day_index]
            else:
                account_indices = []
                trades_skipped += 1

        # === APPLY TRADE ===
        for idx in account_indices:
            balances[idx] += pnl
            trades_executed += 1
            equity_history[idx].append(balances[idx])

            peak[idx] = max(peak[idx], balances[idx])

            freeze_level = START_BALANCE + 1600

            if peak[idx] < freeze_level:
                trailing_floor = peak[idx] - 1500
            else:
                trailing_floor = freeze_level - 1500

            if balances[idx] <= trailing_floor:
                alive[idx] = False

                if first_blow_trade is None:
                    first_blow_trade = i

    total_profit = balances.sum() - NUM_ACCOUNTS * START_BALANCE
    surviving_accounts = alive.sum()
    wiped_out = surviving_accounts == 0

    max_len = max(len(eq) for eq in equity_history)
    for eq in equity_history:
        while len(eq) < max_len:
            eq.append(eq[-1])

    return {
        "total_profit": total_profit,
        "surviving_accounts": surviving_accounts,
        "wiped_out": wiped_out,
        "first_blow_trade": first_blow_trade,
        "trades_executed": trades_executed,
        "trades_skipped": trades_skipped,
        "final_balances": balances,
        "equity_history": equity_history,
        "alive_array": alive
    }


print("\n" + "=" * 70)
print("RUNNING SIMULATIONS...")
print("=" * 70)

results = {}
for mode in ["parallel", "sequential_trade", "one_per_day"]:
    print(f"\n▶ Testing strategy: {mode.replace('_', ' ').title()}")
    results[mode] = simulate(mode)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

for mode, data in results.items():
    print(f"\n📊 {mode.replace('_', ' ').upper()}")
    print("-" * 50)

    # Total profit
    profit = data["total_profit"]
    profit_pct = (profit / (NUM_ACCOUNTS * START_BALANCE)) * 100
    profit_emoji = "✅" if profit > 0 else "❌" if profit < 0 else "⚖️"
    print(f"{profit_emoji} Total P&L: ${profit:+.2f} ({profit_pct:+.1f}%)")

    # Account survival
    survivors = data["surviving_accounts"]
    survival_rate = (survivors / NUM_ACCOUNTS) * 100
    print(f"💪 Accounts surviving: {survivors}/{NUM_ACCOUNTS} ({survival_rate:.1f}%)")

    # Wipeout check
    if data["wiped_out"]:
        print("💀 STATUS: ALL ACCOUNTS WIPED OUT")
    else:
        print("👍 STATUS: Some capital remaining")

    # First blow-up
    if data["first_blow_trade"] is not None:
        first_blow = data["first_blow_trade"]
        print(f"💥 First account blew at trade #{first_blow + 1}")
    else:
        print("✨ No accounts blew up!")

    # Trade execution stats
    exec_rate = (data["trades_executed"] / total_trades) * 100
    print(f"📈 Trades executed: {data['trades_executed']}/{total_trades} ({exec_rate:.1f}%)")
    if data["trades_skipped"] > 0:
        print(f"⏭️ Trades skipped: {data['trades_skipped']}")

    # Final balances per account
    print("\n  Final account balances:")
    for i, bal in enumerate(data["final_balances"]):
        change = bal - START_BALANCE
        status = "ALIVE" if data["alive_array"][i] else "💀 DEAD"
        change_emoji = "🟢" if status == "ALIVE" else "🔴"
        print(f"    Account {i + 1}: ${bal:.2f} ({change:+.2f}) {change_emoji} [{status}]")

print("\n" + "=" * 70)
print("📋 SUMMARY COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame({
    mode: {
        "Total P&L": f"${data['total_profit']:+.2f}",
        "Survivors": f"{data['surviving_accounts']}/{NUM_ACCOUNTS}",
        "Wiped Out?": "YES" if data['wiped_out'] else "no",
        "First Blow": f"Trade #{data['first_blow_trade'] + 1}" if data['first_blow_trade'] is not None else "None",
        "Trades Used": f"{data['trades_executed']}/{total_trades}"
    }
    for mode, data in results.items()
})

print(comparison_df.T.to_string())

# Find the best strategy
profits = {mode: data["total_profit"] for mode, data in results.items()}
best_strategy = max(profits, key=profits.get)
best_profit = profits[best_strategy]

survivors = {mode: data["surviving_accounts"] for mode, data in results.items()}
safest_strategy = max(survivors, key=survivors.get)
safest_survivors = survivors[safest_strategy]

print("\n" + "=" * 70)
print("🏆 RECOMMENDATION")
print("=" * 70)
print(f"Best for profit: {best_strategy.replace('_', ' ').title()} (${best_profit:+.2f})")
print(
    f"Best for safety: {safest_strategy.replace('_', ' ').title()} ({safest_survivors}/{NUM_ACCOUNTS} accounts survive)")
print("=" * 70)

for mode, data in results.items():
    plt.figure()

    for i, eq in enumerate(data["equity_history"]):
        plt.plot(eq)

    plt.title(f"Equity Curves - {mode.replace('_', ' ').title()}")
    plt.xlabel("Trade Index")
    plt.ylabel("Account Balance")

plt.show()
