import pandas as pd
import numpy as np

# ========================================================================================
#  CONFIG
# ========================================================================================
pd.set_option('display.min_rows', 1000)         # Show min 1000 rows when printing
pd.set_option('display.max_rows', 2000)         # Show max 100 rows when printing
pd.set_option('display.max_columns', 10)       # Show max 50 columns when printing

# CSV_PATH = "../CSVS/all_times_14_flat_ONLY_PNL.csv"
# CSV_PATH = "../CSVS/premarket_only.csv"
CSV_PATH = "../CSVS/all_times_14_flat.csv"
START_CAPITAL = 1500

# --- Drawdown settings ---
TRAILING_DD = 1500  # account is closed if DD exceeds this value
DD_FREEZE_TRIGGER = START_CAPITAL + TRAILING_DD + 100
FROZEN_DD_FLOOR = START_CAPITAL + 100
# --- DD stabilization ---
DD_LOOKBACK = 10          # days to check for new lows
REQUIRE_DD_STABLE = False   # require DD to not make new lows in lookback period before starting new account

# --- Date range filter (set to None to disable) ---
START_DATE = None
# START_DATE = "2020-02-01"
# END_DATE = "2020-04-01"
# START_DATE = "2020-01-01"
# END_DATE = "2021-01-20"
END_DATE = None

# --- New account start triggers ---
MAX_ACCOUNTS = 100
# ==================================================================
# --- Profit triggers ---
USE_PROFIT_TRIGGER = False
START_PROFIT_THRESHOLD = 100    # Profit trigger to start next account
END_PROFIT_THRESHOLD = 3000   # Profit level to stop starting new accounts
STEP_PROFIT = 100

# --- Drawdown triggers ---
USE_DD_TRIGGER = True
START_DD_THRESHOLD = 100  # DD trigger to start next account
END_DD_THRESHOLD = 3000    # DD level to stop starting new accounts
STEP_DD = 100

# --- Optimization ranges ---
PROFIT_RANGE = range(START_PROFIT_THRESHOLD, END_PROFIT_THRESHOLD + STEP_PROFIT, STEP_PROFIT)
DD_RANGE = range(START_DD_THRESHOLD, END_DD_THRESHOLD + STEP_DD, STEP_DD)


# --- Recovery requirement ---
RECOVERY_LEVEL = 0   # require DD to recover above this value before next account can start
MIN_DAYS_BETWEEN_STARTS = 1  # minimum days between starting new accounts

# ======================
#  FUNCTIONS
# ======================


def calculate_max_drawdown(equity):
    """Calculate maximum drawdown of an equity series."""
    rolling_max = equity.cummax()
    drawdown = equity - rolling_max
    return drawdown.min()


def build_equity_from_pl(pl_series, offset, st_capital):
    """Start the equity curve from <offset> days later using correct P&L values."""
    shifted_pl = pl_series.iloc[offset:]
    equity = st_capital + shifted_pl.cumsum()
    return equity


def compute_drawdown_series(equity):
    """Return drawdown series indexed by date."""
    rolling_max = equity.cummax()
    drawdown_series = equity - rolling_max
    return drawdown_series


def print_config():
    print("=== Configuration ===")
    print(f"CSV_PATH: {CSV_PATH}")
    print(f"START_CAPITAL: {START_CAPITAL}")
    print(f"TRAILING_DD: {TRAILING_DD}")
    print(f"DD_FREEZE_TRIGGER: {DD_FREEZE_TRIGGER}")
    print(f"FROZEN_DD_FLOOR: {FROZEN_DD_FLOOR}")

    if START_DATE is None:
        print("START_DATE: None (no start date filter)")
    else:
        print(f"START_DATE: {START_DATE}")
    if END_DATE is None:
        print("END_DATE: None (no end date filter)")
    else:
        print(f"END_DATE: {END_DATE}")

    print(f"MAX_ACCOUNTS: {MAX_ACCOUNTS}")
    if USE_DD_TRIGGER:
        print(f"START_IF_DD_THRESHOLD: {START_DD_THRESHOLD}")
    else:
        print("START_IF_DD_THRESHOLD: Disabled")

    if USE_PROFIT_TRIGGER:
        print(f"START_IF_PROFIT_THRESHOLD: {START_PROFIT_THRESHOLD}")
    else:
        print("START_IF_PROFIT_THRESHOLD: Disabled")
    print(f"RECOVERY_LEVEL: {RECOVERY_LEVEL}")
    print(f"MIN_DAYS_BETWEEN_STARTS: {MIN_DAYS_BETWEEN_STARTS}")
    print("=====================")
    if REQUIRE_DD_STABLE:
        print(f"\nREQUIRE_DD_STABLE: {REQUIRE_DD_STABLE}")
        print(f"DD_LOOKBACK: {DD_LOOKBACK} days\n")
    else:
        print(f"\nREQUIRE_DD_STABLE: (disabled)")
        print(f"DD_LOOKBACK: (disabled)\n")


print_config()

# ======================
#  LOAD & CLEAN DATA
# ======================
try:
    df = pd.read_csv(CSV_PATH, sep="\t")
except Exception as e:
    print("Error loading CSV file:".upper(), e)
    exit(1)

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

dd_series = compute_drawdown_series(equity_original)
dd_rolling_min = dd_series.rolling(DD_LOOKBACK, min_periods=1).min()


def run_simulation(dd_threshold, profit_threshold):

    def simulate_staggered_accounts(pl_series, st_capital, max_accounts):
        dates = pl_series.index
        accounts = [{
            'start_idx': 0,
            'equity': st_capital,
            'rolling_max': st_capital,
            'drawdown': 0.0,
            'alive': True
        }]

        # --- START FIRST ACCOUNT RIGHT AWAY ---

        # next_threshold_idx = 0
        last_start_day = 0
        waiting_for_recovery = False

        portfolio_equity = []
        num_acc_alive = []
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
                        dd_floor = acc['rolling_max'] - TRAILING_DD
                    else:
                        # Fixed mode (freeze)
                        dd_floor = FROZEN_DD_FLOOR

                    # Check if account violates DD rule
                    if acc['equity'] <= dd_floor:
                        acc['alive'] = False

                    # Account cannot go below zero
                    if acc['equity'] <= 0:
                        acc['alive'] = False

                today_equities.append(acc['equity'] if acc['start_idx'] <= i_date else np.nan)

            # ----- RECORDING -----
            total_equity = sum(a['equity'] for a in accounts if a['alive'])  # only sum alive accounts
            portfolio_equity.append(total_equity)

            row = [np.nan] * max_accounts
            for k, acc in enumerate(accounts):
                row[k] = acc['equity'] if acc['start_idx'] <= i_date else np.nan
            account_equities_over_time.append(row)

            num_acc_alive.append(sum(a['alive'] for a in accounts))

            # =====================================================
            #   NEW ACCOUNT START LOGIC WITH RECOVERY REQUIREMENT
            # =====================================================

            if len(accounts) < max_accounts:

                # Build drawdown list from alive accounts
                active_dds = [acc['drawdown'] for acc in accounts
                              if acc['alive'] and acc['start_idx'] <= i_date]

                current_dd = min(active_dds) if active_dds else 0

                global_dd_now = dd_series.iloc[i_date]
                global_dd_recent_min = dd_rolling_min.iloc[i_date - 1] if i_date > 0 else 0

                dd_not_making_new_lows = global_dd_now >= global_dd_recent_min

                # ===== START LOGIC =====
                if waiting_for_recovery:
                    can_start = False

                    # Enough recovery?
                    if current_dd >= RECOVERY_LEVEL:
                        waiting_for_recovery = False

                else:
                    trigger_dd = False
                    trigger_profit = False

                    # --- DD TRIGGER ---
                    if USE_DD_TRIGGER and dd_threshold is not None:
                        if current_dd <= -1 * dd_threshold:
                            trigger_dd = True

                    # --- PROFIT TRIGGER (per-account profit) ---
                    if USE_PROFIT_TRIGGER and profit_threshold is not None:

                        # find last alive account
                        alive_accounts = [acc for acc in accounts if acc['alive']]

                        if alive_accounts:
                            last_alive = alive_accounts[-1]
                            acc_profit_since_start = last_alive['equity'] - st_capital

                            if acc_profit_since_start >= profit_threshold:
                                trigger_profit = True
                        else:
                            # if all accounts are blown, allow a new start immediately
                            trigger_profit = True

                    # Combined logic
                    if trigger_dd or trigger_profit:
                        if REQUIRE_DD_STABLE:
                            can_start = dd_not_making_new_lows
                        else:
                            can_start = True

                    else:
                        can_start = False

                # ---- Time-based guard ----
                if can_start and (i_date - last_start_day) >= MIN_DAYS_BETWEEN_STARTS:
                    new_acc = {
                        'start_idx': i_date,
                        'equity': st_capital,
                        'rolling_max': st_capital,
                        'drawdown': 0.0,
                        'alive': True
                    }
                    accounts.append(new_acc)
                    last_start_day = i_date
                    waiting_for_recovery = True  # require recovery before next start
                    # next_threshold_idx += 1

        # Convert to pandas
        portfolio_eq_series = pd.Series(portfolio_equity, index=dates)
        acc_eq_dataframe = pd.DataFrame(account_equities_over_time, index=dates,
                                        columns=[f"acc_{i + 1}" for i in range(max_accounts)])
        num_alive_series = pd.Series(num_acc_alive, index=dates)
        return portfolio_eq_series, acc_eq_dataframe, num_alive_series

    portfolio_eq, acc_eq_df, num_alive = simulate_staggered_accounts(
        pl, START_CAPITAL, MAX_ACCOUNTS
    )

    num_started = acc_eq_df.notna().any().sum()
    num_alive_end = num_alive.iloc[-1]
    final_equity = round(portfolio_eq.iloc[-1], 0)

    # portfolio drawdown
    portfolio_dd = round(compute_drawdown_series(portfolio_eq).min(), 0)

    return {
        "DD_trigger": USE_DD_TRIGGER,
        "Prof_trigger": USE_PROFIT_TRIGGER,
        "DD_thres": dd_threshold,
        "Profit_thres": profit_threshold,
        "Final_equity": final_equity,
        "Accs_started": num_started,
        "Accs_alive": num_alive_end,
        "Accs_blown": num_started - num_alive_end,
        "Portf_max_DD": portfolio_dd
    }


results = []


dd_values = DD_RANGE if USE_DD_TRIGGER else [None]
profit_values = PROFIT_RANGE if USE_PROFIT_TRIGGER else [None]

for dd in dd_values:
    for profit in profit_values:
        res = run_simulation(dd, profit)
        results.append(res)
        print("Done:", res)


results_df = pd.DataFrame(results)

results_df.sort_values(
    by=["Final_equity"],
    ascending=False,
    inplace=True
)

try:
    if USE_PROFIT_TRIGGER:
        file_name_1 = "profit_trigger_optimization_results.xlsx"
    elif USE_DD_TRIGGER:
        file_name_1 = "dd_trigger_optimization_results.xlsx"
    else:
        file_name_1 = "dd_plus_profit_optimization_results.xlsx"

    with pd.ExcelWriter(file_name_1, engine='openpyxl') as writer:
        results_df.to_excel(writer, index=False)
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        # Adjust column widths
        for column_cells in worksheet.columns:
            max_length = max(len(str(cell.value)) for cell in column_cells if cell.value is not None)
            column_letter = column_cells[0].column_letter
            worksheet.column_dimensions[column_letter].width = max_length + 1
    print(f"\nSaved: {file_name_1}")
except Exception as e:
    print("Error saving optimization_results.xlsx:".upper(), e)
