import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np

# ========================================================================================
#  CONFIG
# ========================================================================================
pd.set_option('display.min_rows', 1000)         # Show min 1000 rows when printing
pd.set_option('display.max_rows', 2000)         # Show max 100 rows when printing
pd.set_option('display.max_columns', 10)       # Show max 50 columns when printing

# CSV_PATH = "CSVS/all_times_14_flat_ONLY_PNL.csv"
# CSV_PATH = "CSVS/premarket_only.csv"
CSV_PATH = "CSVS/all_times_14_flat.csv"
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
END_DATE = None

# START_DATE = "2019-10-01"
# # END_DATE = "2021-01-01"
# # START_DATE = "2020-01-01"
# END_DATE = "2021-01-20"


# --- New account start triggers ---
SHOW_PORTFOLIO_TOTAL_EQUITY = True          # if True, show total equity of all accounts combined
SHOW_DRAWDOWN_PLOT = False                  # if True, show drawdown plot

# Which weekday accounts to start (0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday)
# ACTIVE_WEEKDAYS = [0, 1, 2, 3, 4]     # all weekdays
ACTIVE_WEEKDAYS = [4]                   # specific weekdays only (e.g., only Tuesday accounts)


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

    print(f"SHOW_PORTFOLIO_TOTAL_EQUITY: {SHOW_PORTFOLIO_TOTAL_EQUITY}")
    print("=====================")
    if REQUIRE_DD_STABLE:
        print(f"\nREQUIRE_DD_STABLE: {REQUIRE_DD_STABLE}")
        print(f"DD_LOOKBACK: {DD_LOOKBACK} days\n")
    else:
        print(f"\nREQUIRE_DD_STABLE: {REQUIRE_DD_STABLE} (disabled)")
        print(f"DD_LOOKBACK: {DD_LOOKBACK} days (disabled)\n")


print_config()

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

dd_series = compute_drawdown_series(equity_original)
dd_rolling_min = dd_series.rolling(DD_LOOKBACK, min_periods=1).min()


def simulate_weekday_accounts(pl_series, start_capital):
    dates = pl_series.index

    # Weekday mapping: Monday=0 ... Friday=4
    account_weekdays = {
        0: 0,  # acc_1 → Monday
        1: 1,  # acc_2 → Tuesday
        2: 2,  # acc_3 → Wednesday
        3: 3,  # acc_4 → Thursday
        4: 4,  # acc_5 → Friday
    }

    accounts = []
    for wd in ACTIVE_WEEKDAYS:
        accounts.append({
            'weekday': wd,
            'equity': start_capital,
            'rolling_max': start_capital,
            'drawdown': 0.0,
            'alive': True
        })

    portfolio_equity = []
    account_equities_over_time = []
    num_alive = []

    for i_date, date in enumerate(dates):
        weekday = date.weekday()  # Monday=0

        for acc in accounts:
            if not acc['alive']:
                continue

            # Apply P&L only on assigned weekday
            if weekday == acc['weekday']:
                acc['equity'] += pl_series.iloc[i_date]

            # Update rolling max & drawdown
            acc['rolling_max'] = max(acc['rolling_max'], acc['equity'])
            acc['drawdown'] = acc['equity'] - acc['rolling_max']

            # DD floor logic (unchanged)
            if acc['rolling_max'] < DD_FREEZE_TRIGGER:
                dd_floor = acc['rolling_max'] - TRAILING_DD
            else:
                dd_floor = FROZEN_DD_FLOOR

            if acc['equity'] <= dd_floor or acc['equity'] <= 0:
                acc['alive'] = False

        # ----- RECORDING -----
        portfolio_equity.append(sum(a['equity'] for a in accounts if a['alive']))

        row = [acc['equity'] for acc in accounts]
        account_equities_over_time.append(row)

        num_alive.append(sum(a['alive'] for a in accounts))

    portfolio_eq_series = pd.Series(portfolio_equity, index=dates)
    weekday_names = {0: "MO", 1: "TU", 2: "WE", 3: "TH", 4: "FR"}

    acc_eq_df = pd.DataFrame(
        account_equities_over_time,
        index=dates,
        columns=[f"acc_{weekday_names[wd]}" for wd in ACTIVE_WEEKDAYS]
    )
    num_alive_series = pd.Series(num_alive, index=dates)

    return portfolio_eq_series, acc_eq_df, num_alive_series


portfolio_eq, acc_eq_df, num_alive = simulate_weekday_accounts(pl, START_CAPITAL)


# print(f"Acc equity diff{acc_eq_df}")


# ======================
#  DRAWDOWN PLOT
# ======================

if SHOW_DRAWDOWN_PLOT:
    plt.figure(figsize=(10, 5))
    plt.fill_between(dd_series.index, dd_series.values, 0, step="mid", color="blue")
    plt.title("Drawdown Curve")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()

# ======================
#  EQUITY PLOT
# ======================

plt.figure(figsize=(10, 6))
if SHOW_PORTFOLIO_TOTAL_EQUITY:
    plt.plot(portfolio_eq.index, portfolio_eq.values, label="Portfolio total equity", linewidth=4)
for c in acc_eq_df.columns:
    plt.plot(acc_eq_df.index, acc_eq_df[c], alpha=0.8, label=c)
# plt.legend()
plt.title("Staggered Accounts Simulation")
plt.grid(True)
# plt.show()

# quick stats
print("Final portfolio equity:", portfolio_eq.iloc[-1])
print("Num accounts started:", acc_eq_df.notna().any().sum())
print("Number of accounts still alive at end:", num_alive.iloc[-1])

try:
    plt.show()
except KeyboardInterrupt:
    print("Script stopped by user.")
