import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========================================================================================
#  CONFIG
# ========================================================================================

CSV_PATH = "CSVS/premarket_only.csv"
START_CAPITAL = 1500

# START_DATE = "2020-02-24"
# END_DATE = "2021-06-18"
START_DATE = None
END_DATE = None

MAX_ACCOUNTS = 20
# thresholds in money (negative drawdown values where we start next account)
# E.g. start account2 when any active account reaches -600, start 3 at -1000, etc.
START_THRESHOLD = -400  # length = MAX_ACCOUNTS-1
RECOVERY_LEVEL = -0   # require DD to recover above this before next account can start
MIN_DAYS_BETWEEN_STARTS = 30  # optional guard

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


def simulate_staggered_accounts(pl_series, start_capital, max_accounts, threshold):
    dates = pl_series.index
    accounts = []
    next_threshold_idx = 0
    last_start_day = -999

    waiting_for_recovery = False  # <<< NEW

    portfolio_equity = []
    num_alive = []
    account_equities_over_time = []

    for i_date, date in enumerate(dates):

        # ----- UPDATE EXISTING ACCOUNTS -----
        today_equities = []
        for acc in accounts:
            if acc['alive'] and i_date >= acc['start_idx']:

                acc['equity'] += pl_series.iloc[i_date]
                acc['rolling_max'] = max(acc['rolling_max'], acc['equity'])
                acc['drawdown'] = acc['equity'] - acc['rolling_max']

                if acc['equity'] <= 0:
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

            # ---- 1) Always start first account immediately ----
            if len(accounts) == 0:
                can_start = True

            else:
                # Check system-wide max drawdown among active accounts
                current_dd = min(acc['drawdown'] for acc in accounts if acc['start_idx'] <= i_date)

                # (A) If we are still waiting for recovery → do nothing
                if waiting_for_recovery:
                    can_start = False

                    # Check if we have recovered enough
                    if current_dd >= RECOVERY_LEVEL:
                        waiting_for_recovery = False

                # (B) If not waiting for recovery → check threshold trigger
                else:
                    if current_dd <= START_THRESHOLD:
                        can_start = True
                        waiting_for_recovery = True
                    else:
                        can_start = False

            # ---- 2) Time-based guard ----
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


portfolio_eq, acc_eq_df, num_alive = simulate_staggered_accounts(pl, START_CAPITAL, MAX_ACCOUNTS, START_THRESHOLD)

print(f"Acc equity diff{acc_eq_df.head()}")

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

plt.show()
