import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========================================================================================
#  CONFIG
# ========================================================================================
pd.set_option('display.min_rows', 1000)         # Show min 1000 rows when printing
pd.set_option('display.max_rows', 2000)         # Show max 100 rows when printing

CSV_PATH = "CSVS/premarket_only.csv"
# CSV_PATH = "CSVS/all_times_14_flat.csv"
START_CAPITAL = 1500
TRAILING_DD_LIMIT = 1500  # account is closed if DD exceeds this value
DD_FREEZE_TRIGGER = START_CAPITAL + TRAILING_DD_LIMIT + 100
FROZEN_DD_FLOOR = START_CAPITAL + 100

# START_DATE = "2020-03-29"
# END_DATE = "2021-06-18"
START_DATE = None
END_DATE = None

MAX_ACCOUNTS = 20
START_IF_DD_THRESHOLD = -5000  # trigger to start next account
START_IF_PROFIT_THRESHOLD = 1500    # alternative profit trigger to start next account

RECOVERY_LEVEL = -0   # require DD to recover above this before next account can start
MIN_DAYS_BETWEEN_STARTS = 30  # minimum days between starting new accounts

SHOW_PORTFOLIO_TOTAL_EQUITY = False     # if True, show total equity of all accounts combined


# ======================
#  FUNCTIONS
# ======================


def calculate_max_drawdown(equity):
    """Calculate maximum drawdown of an equity series."""
    rolling_max = equity.cummax()
    dd = equity - rolling_max
    return dd.min()


start_capital = START_CAPITAL


def build_equity_from_pl(pl_series, offset, start_capital):
    """Start the equity curve from <offset> days later using correct P&L values."""
    shifted_pl = pl_series.iloc[offset:]
    equity = start_capital + shifted_pl.cumsum()
    return equity


def compute_drawdown_series(equity):
    """Return drawdown series indexed by date."""
    rolling_max = equity.cummax()
    dd_series = equity - rolling_max
    return dd_series


# ======================
#  LOAD & CLEAN DATA
# ======================
df = pd.read_csv(CSV_PATH, sep="\t")
df["P.L"] = df["P.L"].astype(str).str.replace(",", ".").str.strip()
df["P.L"] = pd.to_numeric(df["P.L"], errors='coerce').fillna(0)

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Keep only the necessary columns
df = df[["Date", "P.L"]].dropna()

# Filter by date range
if START_DATE:
    df = df[df["Date"] >= pd.to_datetime(START_DATE)]
if END_DATE:
    df = df[df["Date"] <= pd.to_datetime(END_DATE)]

df = df.sort_values("Date").reset_index(drop=True)

print("Date range:", df["Date"].min(), "→", df["Date"].max())
print("Total rows:", len(df))

# P.L series
pl = df.set_index("Date")["P.L"]

# Original equity curve
equity_original = START_CAPITAL + pl.cumsum()


def simulate_staggered_accounts(pl_series, start_capital, max_accounts):
    dates = pl_series.index
    accounts = []
    # --- START FIRST ACCOUNT RIGHT AWAY ---
    accounts.append({
        'start_idx': 0,
        'equity': start_capital,
        'rolling_max': start_capital,
        'drawdown': 0.0,
        'alive': True
    })

    next_threshold_idx = 0
    last_start_day = 0
    waiting_for_recovery = False

    portfolio_equity = []
    num_alive = []
    account_equities_over_time = []

    for i_date, date in enumerate(dates):

        # ----- UPDATE EXISTING ACCOUNTS -----
        today_equities = []
        for acc in accounts:
            if acc['alive'] and i_date >= acc['start_idx']:
                acc['equity'] += pl_series.iloc[i_date]
                # Update rolling peak
                acc['rolling_max'] = max(acc['rolling_max'], acc['equity'])
                # Compute drawdown
                acc['drawdown'] = acc['equity'] - acc['rolling_max']
                # --- Trailing DD shifts to fixed floor once threshold reached ---
                if acc['rolling_max'] < DD_FREEZE_TRIGGER:
                    # Trailing mode (normal)
                    dd_floor = acc['rolling_max'] - TRAILING_DD_LIMIT
                else:
                    # Fixed mode (freeze)
                    dd_floor = FROZEN_DD_FLOOR

                # Check if account violates DD rule
                if acc['equity'] <= dd_floor:
                    acc['alive'] = False

            today_equities.append(acc['equity'] if acc['start_idx'] <= i_date else np.nan)

        # ----- RECORDING -----
        total_equity = sum(a['equity'] for a in accounts)
        portfolio_equity.append(total_equity)

        row = [np.nan] * max_accounts
        for k, acc in enumerate(accounts):
            row[k] = acc['equity'] if acc['start_idx'] <= i_date else np.nan
        account_equities_over_time.append(row)

        num_alive.append(sum(a['alive'] for a in accounts))

        # =====================================================
        #   NEW ACCOUNT START LOGIC WITH RECOVERY REQUIREMENT
        # =====================================================

        if len(accounts) < max_accounts:

            # Profit since last start
            profit_since_last_start = portfolio_equity[i_date] - portfolio_equity[last_start_day]

            # Build drawdown list from *alive* accounts only
            active_dds = [acc['drawdown'] for acc in accounts
                          if acc['alive'] and acc['start_idx'] <= i_date]

            # If no alive accounts exist, treat drawdown as zero
            if len(active_dds) == 0:
                current_dd = 0
            else:
                current_dd = min(active_dds)

            # ===== START LOGIC =====
            if waiting_for_recovery:
                can_start = False

                # enough recovery?
                if current_dd >= RECOVERY_LEVEL:
                    waiting_for_recovery = False

            else:
                # trigger?
                if current_dd <= START_IF_DD_THRESHOLD or profit_since_last_start >= START_IF_PROFIT_THRESHOLD:
                    can_start = True
                    waiting_for_recovery = True
                else:
                    can_start = False

            # ---- Time-based guard ----
            if can_start and (i_date - last_start_day) >= MIN_DAYS_BETWEEN_STARTS:
                new_acc = {
                    'start_idx': i_date,
                    'equity': start_capital,
                    'rolling_max': start_capital,
                    'drawdown': 0.0,
                    'alive': True
                }

                accounts.append(new_acc)
                last_start_day = i_date
                next_threshold_idx += 1

    # Convert to pandas
    portfolio_eq_series = pd.Series(portfolio_equity, index=dates)
    acc_eq_df = pd.DataFrame(account_equities_over_time, index=dates,
                             columns=[f"acc_{i+1}" for i in range(max_accounts)])
    num_alive_series = pd.Series(num_alive, index=dates)
    return portfolio_eq_series, acc_eq_df, num_alive_series


portfolio_eq, acc_eq_df, num_alive = simulate_staggered_accounts(pl, START_CAPITAL, MAX_ACCOUNTS)

print(f"Acc equity diff{acc_eq_df}")

# plot portfolio and per-account equities
plt.figure(figsize=(14, 6))

if SHOW_PORTFOLIO_TOTAL_EQUITY:
    plt.plot(portfolio_eq.index, portfolio_eq.values, label="Portfolio total equity", linewidth=4)
for c in acc_eq_df.columns:
    plt.plot(acc_eq_df.index, acc_eq_df[c], alpha=0.8, label=c)
plt.legend()
plt.title("Staggered Accounts Simulation")
plt.grid(True)
# plt.show()

# quick stats
print("Final portfolio equity:", portfolio_eq.iloc[-1])
print("Num accounts started:", acc_eq_df.notna().any().sum())
print("Number of accounts still alive at end:", num_alive.iloc[-1])

# ======================
#  DRAWDOWN PLOT
# ======================
dd_series = compute_drawdown_series(equity_original)

plt.figure(figsize=(14, 5))
plt.fill_between(dd_series.index, dd_series.values, 0, step="mid", color="blue")
plt.title("Drawdown Curve")
plt.ylabel("Drawdown")
plt.grid(True)
plt.tight_layout()

try:
    plt.show()
except KeyboardInterrupt:
    print("Script stopped by user.")
