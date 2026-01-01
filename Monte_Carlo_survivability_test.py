import numpy as np

"""
Script to simulate survivability of a trading account using Monte Carlo methods. Doesnt seem realistic
because shows too high survivability. Also BUFFER calculation need to be checked more carefully.
"""
# ==========================
# PARAMETERS (PLAY HERE)
# ==========================
np.random.seed(42)

N_SIM = 50000           # Monte Carlo runs
N_TRADES = 300         # trades per run

WIN_RATE = 0.44

WIN_R = 80.84   # average profitable position in dollars
LOSS_R = -55.02  # average losing position in dollars

TRAILING_STOP = -750    # trailing stop in dollars must be negative
BUFFER = 1800           # try 100, 500, 1200, 1800, 3000

START_SECOND_AT = 0    # trade index when account #2 starts


# ==========================
# TRADE GENERATOR
# ==========================
def generate_trades(n):
    wins = np.random.rand(n) < WIN_RATE
    return np.where(wins, WIN_R, LOSS_R)


# ==========================
# VERBOSE (EDUCATIONAL) SIM
# ==========================
def simulate_account_verbose(trades, buffer):
    equity = 0
    max_equity = 0

    print(f"{'Trade':>5} {'R':>5} {'Equity':>8} {'MaxEq':>8} {'DD':>8}")

    for i, r in enumerate(trades, 1):
        equity += r
        max_equity = max(max_equity, equity)

        if max_equity < buffer:
            trailing_level = TRAILING_STOP + max_equity
        else:
            trailing_level = TRAILING_STOP + buffer

        print(
            f"{i:>5} {r:>5.1f} "
            f"{equity:>8.1f} {max_equity:>8.1f} {trailing_level:>8.1f}"
        )

        if equity <= trailing_level:
            print(">>> ACCOUNT BLOWN <<<\n")
            return False

    print(">>> ACCOUNT SURVIVED <<<\n")
    return True


# ==========================
# FAST (MONTE-CARLO) SIM
# ==========================
def simulate_account(trades, buffer):
    equity = 0
    max_equity = 0

    for r in trades:
        equity += r
        max_equity = max(max_equity, equity)

        if max_equity < buffer:
            trailing_level = TRAILING_STOP + max_equity
        else:
            trailing_level = TRAILING_STOP + buffer

        if equity <= trailing_level:
            return False

    return True


# =====================================================
# 1) VERBOSE DEMO — UNDERSTAND MECHANICS
# =====================================================
print("\n=== VERBOSE EXAMPLE (ONE PATH) ===")
example_trades = generate_trades(1000)
simulate_account_verbose(example_trades, BUFFER)


# =====================================================
# 2) MONTE-CARLO — STATISTICS
# =====================================================
survive_1 = 0
survive_both = 0

for _ in range(N_SIM):
    trades = generate_trades(N_TRADES)

    acc1_ok = simulate_account(trades, BUFFER)

    acc2_ok = True
    if START_SECOND_AT < N_TRADES:
        acc2_ok = simulate_account(trades[START_SECOND_AT:], BUFFER)

    survive_1 += acc1_ok
    survive_both += (acc1_ok and acc2_ok)

print("=== MONTE CARLO RESULTS ===")
print("Survival probability account 1:", survive_1 / N_SIM)
print("Survival probability both:     ", survive_both / N_SIM)
