import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import timedelta

# ========================================================================================
#  CONFIG (same as before)
# ========================================================================================
pd.set_option('display.min_rows', 1000)
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_categories', 10)

CSV_PATH = "../MAE/RG_h1_intervals_night.csv"
START_CAPITAL = 1500

# --- Drawdown settings ---
TRAILING_DD = 1500
DD_FREEZE_TRIGGER = START_CAPITAL + TRAILING_DD + 100
FROZEN_DD_FLOOR = START_CAPITAL + 100

# --- Date range filter ---
START_DATE = "2026-02-09"
END_DATE = None

# ==================================================================
# --- New account start triggers ---
# ==================================================================
MAX_ACCOUNTS = 100
USE_TIME_TRIGGER = True
TIME_TRIGGER_DAYS = 30
USE_PROFIT_TRIGGER = False
START_IF_PROFIT_THRESHOLD = 1000
USE_DD_TRIGGER = False
START_IF_DD_THRESHOLD = 400
RECOVERY_LEVEL = 0
MIN_DAYS_BETWEEN_STARTS = 1

SHOW_PORTFOLIO_TOTAL_PNL = True  # Changed from SHOW_PORTFOLIO_TOTAL_EQUITY
SHOW_DD_PLOT = True
USE_PROP_STYLE_DD = True

# ==================================================================
#  SIMULATION ASSUMPTIONS (IMPORTANT!)
# ==================================================================
"""
KEY ASSUMPTIONS FOR PROP STYLE SIMULATION:

1. Intra-trade sequence: We assume MAE (worst point) occurs BEFORE MFE (best point)
   This is the most conservative assumption for blowout testing.
   In reality, the sequence is unknown - actual results may have more survivors.

2. Capital treatment: Blown accounts have their equity removed entirely from portfolio.
   This models prop firm capital allocation where blown capital is lost.
   For personal account simulation, you'd want to keep the residual capital.

3. Freeze trigger: Once peak reaches DD_FREEZE_TRIGGER, trailing stop freezes permanently.
   This is correctly implemented and persists even if peak later drops.

4. Peak updates: Peak only updates on MFE and profitable exits.
   This matches prop firm rules (trailing from highest equity reached).
"""


# ======================
#  FUNCTIONS
# ======================

def load_and_preprocess_data(csv_path, start_date=None, end_date=None):
    """Load and preprocess the trade data."""
    try:
        df = pd.read_csv(csv_path, sep="\t")
    except Exception as e:
        print("Error loading CSV file:".upper(), e)
        exit(1)

    # Convert columns to numeric
    numeric_columns = ["PNL", "MAE", "MFE"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Convert time columns
    df["Entry_time"] = pd.to_datetime(df["Entry_time"])
    df["Exit_time"] = pd.to_datetime(df["Exit_time"])

    # Filter by date range
    if start_date:
        df = df[df["Exit_time"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Exit_time"] <= pd.to_datetime(end_date)]

    # Sort by entry time
    df = df.sort_values("Entry_time").reset_index(drop=True)

    return df


def compute_prop_style_drawdown(df):
    """
    Compute prop firm style drawdown using MAE/MFE data.
    This creates a floating equity curve that includes intra-trade drawdowns.
    Used for the unified DD plots.
    """
    rows = []
    equity = 0.0

    for _, r in df.sort_values("Entry_time").iterrows():
        # entry
        rows.append({
            "time": r["Entry_time"],
            "equity": equity,
            "event": "entry"
        })

        # worst excursion (MAE)
        rows.append({
            "time": r["Entry_time"],
            "equity": equity + r["MAE"],
            "event": "mae"
        })

        # best excursion (MFE)
        rows.append({
            "time": r["Entry_time"],
            "equity": equity + r["MFE"],
            "event": "mfe"
        })

        # exit
        equity += r["PNL"]
        rows.append({
            "time": r["Exit_time"],
            "equity": equity,
            "event": "exit"
        })

    equity_curve = pd.DataFrame(rows).sort_values("time")

    # Convert time to datetime if not already
    equity_curve["time"] = pd.to_datetime(equity_curve["time"])

    # PROP-FIRM TRAILING DD calculation
    equity_close = 0.0  # cumulative closed PNL
    equity_peak = 0.0  # highest floating equity ever
    worst_dd = 0.0  # worst trailing DD observed

    dd_rows = []

    for _, r in equity_curve.iterrows():
        equity_peak = max(equity_peak, r["equity"])
        dd = r["equity"] - equity_peak
        worst_dd = min(worst_dd, dd)

        dd_rows.append({
            "time": r["time"],
            "equity": r["equity"],
            "equity_peak": equity_peak,
            "trailing_dd": dd,
            "worst_dd_so_far": worst_dd,
            "event": r["event"]
        })

    dd_curve = pd.DataFrame(dd_rows)

    # Prepare daily data for simulation
    daily_data = (
        dd_curve
        .assign(Date=pd.to_datetime(dd_curve["time"]).dt.date)
        .groupby("Date")
        .agg(
            Equity=("equity", "last"),
            Equity_Peak=("equity_peak", "max"),
            Equity_Low=("equity", "min"),
            DD_Floating=("trailing_dd", "min")
        )
        .reset_index()
    )

    daily_data["Date"] = pd.to_datetime(daily_data["Date"])

    # KEEP EVERYTHING IN PURE PNL SPACE
    daily_data["DD_Floating"] = daily_data["Equity_Low"] - daily_data["Equity_Peak"]

    # Also calculate closed DD for comparison
    daily_data["Closed_Peak"] = daily_data["Equity"].cummax()
    daily_data["DD_Closed"] = daily_data["Equity"] - daily_data["Closed_Peak"]

    return daily_data, dd_curve


def create_trade_events_with_priority(df):
    """
    Convert each trade into events with proper priority ordering.

    IMPORTANT ASSUMPTION: We assume MAE occurs before MFE within each trade.
    This is the most conservative assumption for blowout testing.
    In reality, the sequence is unknown and market-dependent.

    The ordering is enforced via priority levels and microsecond offsets:
    - Entry (priority 0)
    - MAE (priority 1) - worst point first (conservative)
    - MFE (priority 2) - best point second
    - Exit (priority 3)
    """
    events = []
    cumulative_pnl = 0

    for trade_idx, trade in df.iterrows():
        # Store pre-trade equity for reference
        pre_trade_equity = cumulative_pnl

        # 1. ENTRY (priority 0)
        events.append({
            'time': trade['Entry_time'],
            'trade_idx': trade_idx,
            'event_type': 'entry',
            'priority': 0,
            'pre_trade_equity': pre_trade_equity,
            'mae': trade['MAE'],
            'mfe': trade['MFE'],
            'pnl': trade['PNL'],
            'equity_change': 0  # No change at entry
        })

        # 2. MAE - worst point (priority 1) - assumed to occur first
        # Microsecond offset ensures ordering when timestamps are identical
        mae_time = trade['Entry_time'] + timedelta(microseconds=1)
        events.append({
            'time': mae_time,
            'trade_idx': trade_idx,
            'event_type': 'mae',
            'priority': 1,
            'pre_trade_equity': pre_trade_equity,
            'mae': trade['MAE'],
            'mfe': trade['MFE'],
            'pnl': trade['PNL'],
            'temp_equity': pre_trade_equity + trade['MAE']  # Temporary probe
        })

        # 3. MFE - best point (priority 2) - assumed to occur after MAE
        mfe_time = trade['Entry_time'] + timedelta(microseconds=2)
        events.append({
            'time': mfe_time,
            'trade_idx': trade_idx,
            'event_type': 'mfe',
            'priority': 2,
            'pre_trade_equity': pre_trade_equity,
            'mae': trade['MAE'],
            'mfe': trade['MFE'],
            'pnl': trade['PNL'],
            'temp_equity': pre_trade_equity + trade['MFE']  # Temporary probe
        })

        # 4. EXIT (priority 3)
        cumulative_pnl += trade['PNL']
        events.append({
            'time': trade['Exit_time'],
            'trade_idx': trade_idx,
            'event_type': 'exit',
            'priority': 3,
            'pre_trade_equity': pre_trade_equity,
            'mae': trade['MAE'],
            'mfe': trade['MFE'],
            'pnl': trade['PNL'],
            'new_equity': cumulative_pnl  # Final equity after trade
        })

    events_df = pd.DataFrame(events)

    # Sort by time AND priority to ensure correct order
    events_df = events_df.sort_values(['time', 'priority']).reset_index(drop=True)

    return events_df


def simulate_accounts_with_prop_dd_optimized(events_df, start_capital, max_accounts):
    """
    OPTIMIZED VERSION: Much faster by using vectorized operations where possible
    and only processing active accounts.
    """
    accounts = []

    # Pre-allocate arrays for faster access
    event_times = events_df['time'].values
    event_types = events_df['event_type'].values
    event_mae = events_df['mae'].values
    event_mfe = events_df['mfe'].values
    event_pnl = events_df['pnl'].values

    total_events = len(events_df)

    # Track account starts - convert to pandas Timestamp for easier date math
    last_start_date = pd.Timestamp(event_times[0])
    waiting_for_recovery = False

    # For fast portfolio tracking - now tracking PNL instead of equity
    portfolio_pnl_history = []
    num_alive_history = []
    portfolio_times = []

    # For account history plotting - we'll sample at exits and blowouts
    account_history_points = []

    # Initialize first account
    accounts.append({
        'id': 1,
        'start_idx': 0,
        'start_date': pd.Timestamp(event_times[0]),
        'equity': start_capital,
        'pnl': 0,  # Track PNL separately
        'peak': start_capital,
        'alive': True,
        'current_trade_start_equity': None,
        'freeze_triggered': False,
        'last_event_idx': 0  # Track last processed event for this account
    })

    # Main simulation loop
    for event_idx in range(total_events):
        current_time = pd.Timestamp(event_times[event_idx])
        event_type = event_types[event_idx]

        # Process only accounts that started before or at this event and are still alive
        # Using list comprehension is faster than filtering in the loop
        active_accounts = []
        for acc in accounts:
            if acc['alive'] and acc['start_idx'] <= event_idx:
                active_accounts.append(acc)

        for acc in active_accounts:
            # Skip if this account already processed this event
            if acc['last_event_idx'] >= event_idx:
                continue

            acc['last_event_idx'] = event_idx

            # Store pre-event equity
            pre_event_equity = acc['equity']

            if event_type == 'entry':
                # Start of a new trade
                acc['current_trade_start_equity'] = acc['equity']

            elif event_type == 'mae':
                # Check MAE blowout
                if acc['current_trade_start_equity'] is not None:
                    temp_equity = acc['current_trade_start_equity'] + event_mae[event_idx]

                    if acc['peak'] < DD_FREEZE_TRIGGER:
                        dd_floor = acc['peak'] - TRAILING_DD
                    else:
                        dd_floor = FROZEN_DD_FLOOR
                        acc['freeze_triggered'] = True

                    if temp_equity <= dd_floor:
                        acc['alive'] = False
                        print(f"Account {acc['id']} BLOWN at MAE on {current_time}")
                        # Record blowout for history
                        account_history_points.append({
                            'time': current_time,
                            'account_id': acc['id'],
                            'equity': pre_event_equity,
                            'pnl': acc['pnl'],
                            'event': 'blowout_mae'
                        })

            elif event_type == 'mfe':
                # Update peak if MFE creates new high
                if acc['current_trade_start_equity'] is not None and acc['alive']:
                    temp_equity = acc['current_trade_start_equity'] + event_mfe[event_idx]
                    if temp_equity > acc['peak']:
                        acc['peak'] = temp_equity

            elif event_type == 'exit':
                # Process exit
                if acc['current_trade_start_equity'] is not None and acc['alive']:
                    acc['equity'] = acc['current_trade_start_equity'] + event_pnl[event_idx]
                    acc['pnl'] = acc['equity'] - start_capital  # Update PNL

                    if acc['equity'] > acc['peak']:
                        acc['peak'] = acc['equity']
                        if acc['peak'] >= DD_FREEZE_TRIGGER:
                            acc['freeze_triggered'] = True

                    acc['current_trade_start_equity'] = None

                    # Check blowout at exit
                    if acc['peak'] < DD_FREEZE_TRIGGER:
                        dd_floor = acc['peak'] - TRAILING_DD
                    else:
                        dd_floor = FROZEN_DD_FLOOR

                    if acc['equity'] <= dd_floor:
                        acc['alive'] = False
                        print(f"Account {acc['id']} BLOWN at exit on {current_time}")
                        account_history_points.append({
                            'time': current_time,
                            'account_id': acc['id'],
                            'equity': acc['equity'],
                            'pnl': acc['pnl'],
                            'event': 'blowout_exit'
                        })
                    else:
                        # Record normal exit for history
                        account_history_points.append({
                            'time': current_time,
                            'account_id': acc['id'],
                            'equity': acc['equity'],
                            'pnl': acc['pnl'],
                            'event': 'exit'
                        })

        # Record portfolio state at exits and blowouts - now tracking PNL
        if event_type in ['exit', 'mae']:
            total_pnl = sum(acc['pnl'] for acc in accounts if acc['alive'])
            alive_count = sum(1 for acc in accounts if acc['alive'])

            portfolio_pnl_history.append(total_pnl)
            num_alive_history.append(alive_count)
            portfolio_times.append(current_time)

        # Start new accounts at exits
        if event_type == 'exit' and len(accounts) < max_accounts:

            # Calculate current drawdown for triggers
            alive_accounts = [acc for acc in accounts if acc['alive']]
            if alive_accounts:
                current_dd = min([acc['equity'] - acc['peak'] for acc in alive_accounts])
            else:
                current_dd = 0

            can_start = False
            started_due_to_dd = False

            if waiting_for_recovery and USE_DD_TRIGGER:
                if current_dd >= RECOVERY_LEVEL:
                    waiting_for_recovery = False
                else:
                    can_start = False

            if not waiting_for_recovery:
                trigger_dd = False
                trigger_profit = False
                trigger_time = False

                if USE_DD_TRIGGER and START_IF_DD_THRESHOLD:
                    if current_dd <= -START_IF_DD_THRESHOLD:
                        trigger_dd = True
                        started_due_to_dd = True

                if USE_PROFIT_TRIGGER and START_IF_PROFIT_THRESHOLD and alive_accounts:
                    last_alive = alive_accounts[-1]
                    if last_alive['equity'] - start_capital >= START_IF_PROFIT_THRESHOLD:
                        trigger_profit = True

                if USE_TIME_TRIGGER:
                    # Fix: Use total_seconds() to calculate days difference
                    time_diff = current_time - last_start_date
                    days_since_last = time_diff.total_seconds() / (24 * 3600)
                    if days_since_last >= TIME_TRIGGER_DAYS:
                        trigger_time = True

                if trigger_dd or trigger_profit or trigger_time:
                    can_start = True

            if can_start:
                # Fix: Calculate days difference for MIN_DAYS_BETWEEN_STARTS
                time_diff = current_time - last_start_date
                days_since_last = time_diff.total_seconds() / (24 * 3600)
                if days_since_last >= MIN_DAYS_BETWEEN_STARTS:
                    accounts.append({
                        'id': len(accounts) + 1,
                        'start_idx': event_idx,
                        'start_date': current_time,
                        'equity': start_capital,
                        'pnl': 0,
                        'peak': start_capital,
                        'alive': True,
                        'freeze_triggered': False,
                        'current_trade_start_equity': None,
                        'last_event_idx': event_idx
                    })
                    print(f"Started Account {len(accounts)} on {current_time}")
                    last_start_date = current_time
                    waiting_for_recovery = USE_DD_TRIGGER and started_due_to_dd

    print(f"\nSimulation complete. Processed {total_events} events, {len(accounts)} accounts.")

    # Create DataFrames from recorded history - now tracking PNL
    if portfolio_times:
        portfolio_pnl_series = pd.Series(portfolio_pnl_history, index=portfolio_times, name='portfolio_pnl')
        portfolio_pnl_daily = portfolio_pnl_series.resample('D').last().fillna(method='ffill')

        num_alive_series = pd.Series(num_alive_history, index=portfolio_times, name='num_alive')
        num_alive_daily = num_alive_series.resample('D').last().fillna(method='ffill')
    else:
        portfolio_pnl_daily = pd.Series()
        num_alive_daily = pd.Series()

    # Create account PNL DataFrame from history points
    if account_history_points:
        history_df = pd.DataFrame(account_history_points)
        # Pivot to get account PNL over time
        account_pnl = history_df.pivot_table(
            index='time',
            columns='account_id',
            values='pnl',
            aggfunc='last'
        )
        account_pnl.columns = [f'acc_{col}_pnl' for col in account_pnl.columns]
        account_pnl = account_pnl.resample('D').last().fillna(method='ffill')
    else:
        account_pnl = pd.DataFrame()

    return portfolio_pnl_daily, account_pnl, num_alive_daily, accounts


def simulate_accounts_closed_dd(pl_series, start_capital, max_accounts):
    """Original closed-equity simulation (simplified)."""
    dates = pl_series.index
    accounts = []

    # Start first account
    accounts.append({
        'id': 1,
        'start_idx': 0,
        'start_date': dates[0],
        'equity': start_capital,
        'pnl': 0,
        'rolling_max': start_capital,
        'alive': True
    })

    last_start_date = dates[0]
    portfolio_pnl = []
    num_alive = []
    account_pnls = []

    for i_date, date in enumerate(dates):
        # Update accounts
        for acc in accounts:
            if acc['alive'] and i_date >= acc['start_idx']:
                acc['equity'] += pl_series.iloc[i_date]
                acc['pnl'] = acc['equity'] - start_capital
                acc['rolling_max'] = max(acc['rolling_max'], acc['equity'])

                # Check blowout
                if acc['rolling_max'] < DD_FREEZE_TRIGGER:
                    dd_floor = acc['rolling_max'] - TRAILING_DD
                else:
                    dd_floor = FROZEN_DD_FLOOR

                if acc['equity'] <= dd_floor:
                    acc['alive'] = False

        # Record - now tracking PNL
        total_pnl = sum(acc['pnl'] for acc in accounts if acc['alive'])
        portfolio_pnl.append(total_pnl)
        num_alive.append(sum(1 for acc in accounts if acc['alive']))

        row = {f'acc_{acc["id"]}_pnl': acc['pnl'] if acc['start_idx'] <= i_date else np.nan
               for acc in accounts}
        row['time'] = date
        account_pnls.append(row)

        # Start new accounts (time-based only for closed mode)
        if len(accounts) < max_accounts and USE_TIME_TRIGGER:
            if (date - last_start_date).days >= TIME_TRIGGER_DAYS:
                accounts.append({
                    'id': len(accounts) + 1,
                    'start_idx': i_date,
                    'start_date': date,
                    'equity': start_capital,
                    'pnl': 0,
                    'rolling_max': start_capital,
                    'alive': True
                })
                last_start_date = date

    portfolio_pnl_series = pd.Series(portfolio_pnl, index=dates, name='portfolio_pnl')
    accounts_df = pd.DataFrame(account_pnls).set_index('time')
    num_alive_series = pd.Series(num_alive, index=dates, name='num_alive')

    return portfolio_pnl_series, accounts_df, num_alive_series, accounts


def print_config():
    print("=== Configuration ===")
    print(f"CSV_PATH: {CSV_PATH}")
    print(f"START_CAPITAL: {START_CAPITAL}")
    print(f"TRAILING_DD: {TRAILING_DD}")
    print(f"DD_FREEZE_TRIGGER: {DD_FREEZE_TRIGGER}")
    print(f"FROZEN_DD_FLOOR: {FROZEN_DD_FLOOR}")
    print(f"USE_PROP_STYLE_DD: {USE_PROP_STYLE_DD}")
    if START_DATE:
        print(f"START_DATE: {START_DATE}")
    if END_DATE:
        print(f"END_DATE: {END_DATE}")
    print(f"MAX_ACCOUNTS: {MAX_ACCOUNTS}")
    print(f"TIME_TRIGGER_DAYS: {TIME_TRIGGER_DAYS}")
    print("=====================")


# ======================
#  MAIN EXECUTION
# ======================
print_config()

# Load data
df = load_and_preprocess_data(CSV_PATH, START_DATE, END_DATE)
print(f"\nLoaded {len(df)} trades from {df['Entry_time'].min()} to {df['Exit_time'].max()}")

# For the unified DD plots, we need the prop-style drawdown data
if USE_PROP_STYLE_DD:
    print("\nComputing prop-style drawdown for visualization...")
    daily_data, full_dd_curve = compute_prop_style_drawdown(df)
    plot_df = daily_data.copy()

    # Also create daily P&L series for monthly/yearly charts
    daily_pnl_for_plots = daily_data[["Date", "Equity"]].copy()
    daily_pnl_for_plots.rename(columns={"Equity": "PNL_Daily"}, inplace=True)

    # Calculate P&L correctly
    daily_pnl_for_plots["PNL_Daily"] = daily_pnl_for_plots["PNL_Daily"].diff()
    # First day P&L is equity - START_CAPITAL
    if len(daily_pnl_for_plots) > 0:
        daily_pnl_for_plots.iloc[0, daily_pnl_for_plots.columns.get_loc('PNL_Daily')] = \
            daily_data["Equity"].iloc[0]

    daily_pnl_for_plots.set_index('Date', inplace=True)

else:
    # For closed mode, create daily P&L from trades directly
    df['Date'] = df['Exit_time'].dt.date
    daily_pnl = df.groupby('Date')['PNL'].sum()
    daily_pnl.index = pd.to_datetime(daily_pnl.index)
    daily_pnl = daily_pnl.sort_index()

    # For days with no trades, P&L is 0
    # Create a complete date range
    if len(daily_pnl) > 0:
        date_range = pd.date_range(start=daily_pnl.index.min(), end=daily_pnl.index.max(), freq='D')
        daily_pnl = daily_pnl.reindex(date_range, fill_value=0)

    # Create plot_df for closed mode
    equity_original = START_CAPITAL + daily_pnl.cumsum()
    dd_series = equity_original - equity_original.cummax()

    plot_df = pd.DataFrame({
        "Date": dd_series.index,
        "Equity": equity_original.values,
        "Equity_Peak": equity_original.cummax().values,
        "Equity_Low": equity_original.values,
        "DD_Closed": dd_series.values,
        "DD_Floating": dd_series.values
    })

    daily_pnl_for_plots = daily_pnl.to_frame(name='PNL_Daily')

print("\n" + "=" * 60)
print("SIMULATION MODE:",
      "Prop Firm (Intraday MAE-based)" if USE_PROP_STYLE_DD else "Closed Equity (Daily close-based)")
print("=" * 60)

if USE_PROP_STYLE_DD:
    print("\nIMPORTANT ASSUMPTIONS:")
    print("1. Intra-trade sequence: MAE (worst) → MFE (best) → Exit")
    print("   This is the most conservative assumption for blowout testing.")
    print("2. Capital treatment: Blown accounts lose all equity (prop firm model)")
    print("3. Freeze trigger: Once peak reaches threshold, stop freezes permanently")
    print("=" * 60)

    # Create event sequence with proper ordering
    events_df = create_trade_events_with_priority(df)
    print(f"Created {len(events_df)} events from {len(df)} trades")
    print(f"Event type distribution:")
    print(events_df['event_type'].value_counts())

    # Run OPTIMIZED simulation
    print("\nRunning simulation (optimized)...")
    portfolio_pnl, acc_pnl_df, num_alive_df, accounts = simulate_accounts_with_prop_dd_optimized(
        events_df, START_CAPITAL, MAX_ACCOUNTS
    )

else:
    # Run closed equity simulation
    portfolio_pnl, acc_pnl_df, num_alive_df, accounts = simulate_accounts_closed_dd(
        daily_pnl, START_CAPITAL, MAX_ACCOUNTS
    )

# ============================================================
# UNIFIED EQUITY + DD PLOTS (Same Style as Original Script)
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# ============================================
# 1) Equity Curve
# ============================================
axes[0].plot(plot_df["Date"], plot_df["Equity"], linewidth=2, label="Equity")
axes[0].plot(plot_df["Date"], plot_df["Equity_Peak"], linewidth=1, label="Equity_Peak")
axes[0].plot(plot_df["Date"], plot_df["Equity_Low"], linewidth=1, label="Equity_Low")
axes[0].set_title("Equity Curve (Single Account Strategy)")
axes[0].set_ylabel("Equity ($)")
axes[0].grid(True)
axes[0].legend()

# ============================================
# 2) Closed DD
# ============================================
axes[1].plot(plot_df["Date"], plot_df["DD_Closed"], linewidth=2, label="Closed DD")
axes[1].axhline(0, linewidth=0.8)
axes[1].set_title("Closed Equity Drawdown")
axes[1].set_ylabel("Drawdown ($)")
axes[1].grid(True)
axes[1].legend()

# ============================================
# 3) Floating DD
# ============================================
axes[2].plot(plot_df["Date"], plot_df["DD_Floating"], linewidth=2, label="Floating DD")
axes[2].axhline(0, linewidth=0.8)
axes[2].set_title("Floating Drawdown (Prop Style - Includes Intraday)")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Drawdown ($)")
axes[2].grid(True)
axes[2].legend()

# Force every month on x-axis for shared equity/DD plots
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()

# ======================
# PORTFOLIO PNL PLOT (replacing Portfolio Total Equity)
# ======================

if SHOW_PORTFOLIO_TOTAL_PNL and not portfolio_pnl.empty:
    fig_portfolio, ax_portfolio = plt.subplots(figsize=(14, 6))

    ax_portfolio.plot(
        portfolio_pnl.index,
        portfolio_pnl.values,
        linewidth=3,
        color='blue',
        label="Portfolio Total P&L"
    )

    # Add zero line for reference
    ax_portfolio.axhline(y=0, color='black', linewidth=0.8, alpha=0.5, linestyle='--')

    ax_portfolio.set_title("Portfolio Total P&L (All Accounts Combined)")
    ax_portfolio.set_ylabel("P&L ($)")
    ax_portfolio.set_xlabel("Date")
    ax_portfolio.grid(True, alpha=0.3)
    ax_portfolio.legend()

    # Format x-axis to show months nicely
    ax_portfolio.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_portfolio.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax_portfolio.xaxis.get_majorticklabels(), rotation=45)

    plt.setp(ax_portfolio.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()

# ======================
# INDIVIDUAL ACCOUNTS P&L PLOT (now showing P&L instead of equity)
# ======================

if not acc_pnl_df.empty:
    fig_accounts, ax_accounts = plt.subplots(figsize=(14, 6))

    number_accounts_started = len(accounts)

    # Plot each account's P&L curve
    for i, col in enumerate(acc_pnl_df.columns[:number_accounts_started]):
        ax_accounts.plot(
            acc_pnl_df.index,
            acc_pnl_df[col],
            alpha=0.7,
            linewidth=1.5,
            label=f"Account {i + 1}" if i < 10 else None  # Only label first 10 for legend
        )

    ax_accounts.set_title("Individual Accounts P&L")
    ax_accounts.set_ylabel("P&L ($)")
    ax_accounts.set_xlabel("Date")
    ax_accounts.grid(True, alpha=0.3)
    ax_accounts.axhline(y=0, color='black', linewidth=0.8, alpha=0.5, linestyle='--')

    # Format x-axis to show months nicely
    ax_accounts.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_accounts.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax_accounts.xaxis.get_majorticklabels(), rotation=45)

    # Frozen DD floor line
    ax_accounts.axhline(
        y=FROZEN_DD_FLOOR,
        color='red',
        linewidth=2,
        linestyle='--',
        alpha=0.8,
        label=f'Frozen DD floor line{FROZEN_DD_FLOOR}'
    )

    plt.setp(ax_accounts.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()

# ======================
# ACCOUNT STATUS OVER TIME
# ======================

if not num_alive_df.empty:
    fig3, ax5 = plt.subplots(1, 1, figsize=(14, 6))

    # Plot number of alive accounts over time
    ax5.plot(num_alive_df.index, num_alive_df.values,
             color="purple", linewidth=3, label="Number of Alive Accounts")
    ax5.fill_between(num_alive_df.index, num_alive_df.values, 0,
                     color="purple", alpha=0.2)
    ax5.set_title("Number of Active Accounts Over Time")
    ax5.set_ylabel("Number of Accounts")
    ax5.set_xlabel("Date")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Add horizontal lines for reference
    ax5.axhline(y=number_accounts_started, color='gray', linestyle='--',
                alpha=0.5, label=f"Total Started: {number_accounts_started}")
    ax5.axhline(y=num_alive_df.iloc[-1], color='green', linestyle='--',
                alpha=0.5, label=f"Final Alive: {num_alive_df.iloc[-1]}")

    ax5.legend(loc='upper left')

    # Rotate x-axis labels
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()


# ============================================================
# CHART 0: DAILY P&L (Single Account Strategy)
# ============================================================

daily_pnl_series = daily_pnl_for_plots['PNL_Daily']

fig_daily, ax_daily = plt.subplots(figsize=(16, 6))

colors = ['green' if x >= 0 else 'red' for x in daily_pnl_series.values]

bars = ax_daily.bar(
    daily_pnl_series.index,
    daily_pnl_series.values,
    color=colors,
    alpha=0.7,
    edgecolor='black',
    linewidth=0.3
)

# Zero line
ax_daily.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)

# X-axis formatting (daily)
ax_daily.xaxis.set_major_locator(mdates.DayLocator(interval=5))   # every 5 days (adjust if needed)
ax_daily.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.setp(ax_daily.xaxis.get_majorticklabels(), rotation=60)

# Y-axis formatting (daily)
ax_daily.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
ax_daily.yaxis.set_major_locator(mticker.MultipleLocator(50))

# Titles / labels
ax_daily.set_title("Single Account Strategy - Daily P&L", fontsize=14, fontweight='bold')
ax_daily.set_ylabel("P&L ($)")
ax_daily.set_xlabel("Date")
ax_daily.grid(True, alpha=0.3, axis='y')

# Summary stats
total_daily = daily_pnl_series.sum()
positive_days = (daily_pnl_series > 0).sum()
win_rate_daily = positive_days / len(daily_pnl_series) * 100 if len(daily_pnl_series) > 0 else 0

textstr = f'Total: ${total_daily:,.0f} | Win Rate: {win_rate_daily:.1f}% ({positive_days}/{len(daily_pnl_series)})'
ax_daily.text(0.02, 0.98, textstr, transform=ax_daily.transAxes,
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# ============================================================
# CHART 1: MONTHLY P&L (Single Account Strategy)
# ============================================================

# Group by month and sum P&L
monthly_pnl = daily_pnl_for_plots.resample('M')['PNL_Daily'].sum()

# Create monthly P&L bar chart
fig_monthly, ax_monthly = plt.subplots(figsize=(14, 6))

# Define colors: green for positive, red for negative
colors = ['green' if x >= 0 else 'red' for x in monthly_pnl.values]

# Create bar chart
bars = ax_monthly.bar(monthly_pnl.index, monthly_pnl.values,
                      color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

# Add a horizontal line at y=0
ax_monthly.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)

# Format x-axis to show months nicely
ax_monthly.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax_monthly.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax_monthly.xaxis.get_majorticklabels(), rotation=45)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    if height >= 0:
        label_position = height + (monthly_pnl.max() * 0.01)
    else:
        label_position = height - (monthly_pnl.max() * 0.01)

    ax_monthly.text(bar.get_x() + bar.get_width() / 2., label_position,
                    f'${height:,.0f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8, rotation=45)

# Set titles and labels
ax_monthly.set_title("Single Account Strategy - Monthly P&L", fontsize=14, fontweight='bold')
ax_monthly.set_ylabel("P&L ($)")
ax_monthly.set_xlabel("Date")
ax_monthly.grid(True, alpha=0.3, axis='y')

# Add summary statistics
total_pnl = monthly_pnl.sum()
positive_months = (monthly_pnl > 0).sum()
negative_months = (monthly_pnl < 0).sum()
win_rate = positive_months / len(monthly_pnl) * 100 if len(monthly_pnl) > 0 else 0

textstr = f'Total: ${total_pnl:,.0f} | Win Rate: {win_rate:.1f}% ({positive_months}/{len(monthly_pnl)})'
ax_monthly.text(0.02, 0.98, textstr, transform=ax_monthly.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# ============================================================
# CHART 2: YEARLY P&L (Single Account Strategy)
# ============================================================

# Group by year and sum P&L
yearly_pnl = daily_pnl_for_plots.resample('Y')['PNL_Daily'].sum()

# Create yearly P&L bar chart
fig_yearly, ax_yearly = plt.subplots(figsize=(12, 6))

# Define colors: green for positive, red for negative
colors = ['green' if x >= 0 else 'red' for x in yearly_pnl.values]

# Create bar chart
bars = ax_yearly.bar(yearly_pnl.index.year, yearly_pnl.values,
                     color=colors, alpha=0.7, edgecolor='black', linewidth=0.8, width=0.6)

# Add a horizontal line at y=0
ax_yearly.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    if height >= 0:
        label_position = height + (yearly_pnl.max() * 0.02)
    else:
        label_position = height - (yearly_pnl.max() * 0.02)

    ax_yearly.text(bar.get_x() + bar.get_width() / 2., label_position,
                   f'${height:,.0f}',
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10, fontweight='bold')

# Set titles and labels
ax_yearly.set_title("Single Account Strategy - Yearly P&L", fontsize=14, fontweight='bold')
ax_yearly.set_ylabel("P&L ($)")
ax_yearly.set_xlabel("Year")
ax_yearly.grid(True, alpha=0.3, axis='y')

# Add summary statistics
total_pnl_yearly = yearly_pnl.sum()
avg_yearly = yearly_pnl.mean() if len(yearly_pnl) > 0 else 0
positive_years = (yearly_pnl > 0).sum()
negative_years = (yearly_pnl < 0).sum()
win_rate_yearly = positive_years / len(yearly_pnl) * 100 if len(yearly_pnl) > 0 else 0

textstr = f'Total: ${total_pnl_yearly:,.0f} | Avg: ${avg_yearly:,.0f} | Win Rate: {win_rate_yearly:.1f}% ({positive_years}/{len(yearly_pnl)})'
ax_yearly.text(0.02, 0.98, textstr, transform=ax_yearly.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# ============================================================
# CHART 3: PORTFOLIO MONTHLY P&L (now using portfolio_pnl)
# ============================================================

if not portfolio_pnl.empty:
    # Calculate daily portfolio P&L from portfolio PNL curve
    portfolio_daily_pnl = portfolio_pnl.diff().fillna(portfolio_pnl.iloc[0])
    portfolio_daily_pnl.name = 'Portfolio_PNL_Daily'

    # Group by month and sum portfolio P&L
    portfolio_monthly_pnl = portfolio_daily_pnl.resample('M').sum()

    # Create portfolio monthly P&L bar chart
    fig_portfolio_monthly, ax_portfolio_monthly = plt.subplots(figsize=(14, 6))

    # Define colors: green for positive, red for negative
    colors = ['green' if x >= 0 else 'red' for x in portfolio_monthly_pnl.values]

    # Create bar chart
    bars = ax_portfolio_monthly.bar(portfolio_monthly_pnl.index, portfolio_monthly_pnl.values,
                                    color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add a horizontal line at y=0
    ax_portfolio_monthly.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)

    # Format x-axis to show months nicely
    ax_portfolio_monthly.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_portfolio_monthly.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax_portfolio_monthly.xaxis.get_majorticklabels(), rotation=45)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height >= 0:
            label_position = height + (portfolio_monthly_pnl.max() * 0.01)
        else:
            label_position = height - (portfolio_monthly_pnl.max() * 0.01)

        ax_portfolio_monthly.text(bar.get_x() + bar.get_width() / 2., label_position,
                                  f'${height:,.0f}',
                                  ha='center', va='bottom' if height >= 0 else 'top',
                                  fontsize=8, rotation=45)

    # Set titles and labels
    ax_portfolio_monthly.set_title("Portfolio (All Accounts) - Monthly P&L", fontsize=14, fontweight='bold')
    ax_portfolio_monthly.set_ylabel("P&L ($)")
    ax_portfolio_monthly.set_xlabel("Date")
    ax_portfolio_monthly.grid(True, alpha=0.3, axis='y')

    # Add summary statistics
    portfolio_total_pnl = portfolio_monthly_pnl.sum()
    portfolio_positive_months = (portfolio_monthly_pnl > 0).sum()
    portfolio_negative_months = (portfolio_monthly_pnl < 0).sum()
    portfolio_win_rate = portfolio_positive_months / len(portfolio_monthly_pnl) * 100 if len(
        portfolio_monthly_pnl) > 0 else 0

    textstr = f'Total: ${portfolio_total_pnl:,.0f} | Win Rate: {portfolio_win_rate:.1f}% ({portfolio_positive_months}/{len(portfolio_monthly_pnl)})'
    ax_portfolio_monthly.text(0.02, 0.98, textstr, transform=ax_portfolio_monthly.transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

# ============================================================
# CHART 4: PORTFOLIO YEARLY P&L (now using portfolio_pnl)
# ============================================================

if not portfolio_pnl.empty:
    # Group by year and sum portfolio P&L
    portfolio_yearly_pnl = portfolio_daily_pnl.resample('Y').sum()

    # Create portfolio yearly P&L bar chart
    fig_portfolio_yearly, ax_portfolio_yearly = plt.subplots(figsize=(12, 6))

    # Define colors: green for positive, red for negative
    colors = ['green' if x >= 0 else 'red' for x in portfolio_yearly_pnl.values]

    # Create bar chart
    bars = ax_portfolio_yearly.bar(portfolio_yearly_pnl.index.year, portfolio_yearly_pnl.values,
                                   color=colors, alpha=0.7, edgecolor='black', linewidth=0.8, width=0.6)

    # Add a horizontal line at y=0
    ax_portfolio_yearly.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height >= 0:
            label_position = height + (portfolio_yearly_pnl.max() * 0.02)
        else:
            label_position = height - (portfolio_yearly_pnl.max() * 0.02)

        ax_portfolio_yearly.text(bar.get_x() + bar.get_width() / 2., label_position,
                                 f'${height:,.0f}',
                                 ha='center', va='bottom' if height >= 0 else 'top',
                                 fontsize=10, fontweight='bold')

    # Set titles and labels
    ax_portfolio_yearly.set_title("Portfolio (All Accounts) - Yearly P&L", fontsize=14, fontweight='bold')
    ax_portfolio_yearly.set_ylabel("P&L ($)")
    ax_portfolio_yearly.set_xlabel("Year")
    ax_portfolio_yearly.grid(True, alpha=0.3, axis='y')

    # Add summary statistics
    portfolio_total_pnl_yearly = portfolio_yearly_pnl.sum()
    portfolio_avg_yearly = portfolio_yearly_pnl.mean() if len(portfolio_yearly_pnl) > 0 else 0
    portfolio_positive_years = (portfolio_yearly_pnl > 0).sum()
    portfolio_negative_years = (portfolio_yearly_pnl < 0).sum()
    portfolio_win_rate_yearly = portfolio_positive_years / len(portfolio_yearly_pnl) * 100 if len(
        portfolio_yearly_pnl) > 0 else 0

    textstr = f'Total: ${portfolio_total_pnl_yearly:,.0f} | Avg: ${portfolio_avg_yearly:,.0f} | Win Rate: {portfolio_win_rate_yearly:.1f}% ({portfolio_positive_years}/{len(portfolio_yearly_pnl)})'
    ax_portfolio_yearly.text(0.02, 0.98, textstr, transform=ax_portfolio_yearly.transAxes,
                             fontsize=10, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

# ======================
#  STATISTICS
# ======================
print("\n" + "=" * 60)
print("SIMULATION RESULTS")
print("=" * 60)

print("\nFINAL P&L PER ACCOUNT")
print("-" * 60)

for acc in accounts:
    status = "ALIVE" if acc['alive'] else "BLOWN"
    print(
        f"Account {acc['id']:>2} | "
        f"Status: {status:<5} | "
        f"Final P&L: ${acc['pnl']:>8.2f}"
    )

print("-" * 60)

number_accounts_started = len(accounts)
number_accounts_alive = sum(1 for acc in accounts if acc['alive'])
final_portfolio_pnl = portfolio_pnl.iloc[-1] if not portfolio_pnl.empty else 0
total_capital_deployed = START_CAPITAL * number_accounts_started

print(f"{'Simulation Mode:':<30} {'Prop Firm (Intraday)' if USE_PROP_STYLE_DD else 'Closed Equity'}")
print(f"{'Final Portfolio P&L:':<30} ${final_portfolio_pnl:,.2f}")
print(f"{'Total Capital Deployed:':<30} ${total_capital_deployed:,.2f}")
print(f"{'Return on Capital:':<30} {(final_portfolio_pnl / total_capital_deployed * 100):.1f}%")
print(f"{'Accounts Started:':<30} {number_accounts_started}")
print(f"{'Accounts Alive:':<30} {number_accounts_alive}")
print(f"{'Accounts Blown:':<30} {number_accounts_started - number_accounts_alive}")
print(f"{'Survival Rate:':<30} {(number_accounts_alive / number_accounts_started * 100):.1f}%")

if USE_PROP_STYLE_DD:
    freeze_triggered_count = 0

    for acc in accounts:
        if acc['freeze_triggered']:
            freeze_triggered_count += 1

    print(f"\nBlowout Analysis:")
    print(f"{'Accounts that hit freeze trigger:':<30} {freeze_triggered_count}")
    print(f"{'Total Blown Accounts:':<30} {number_accounts_started - number_accounts_alive}")

print("\n" + "-" * 60)
print("SINGLE ACCOUNT STRATEGY PERFORMANCE")
print("-" * 60)

# Ensure monthly_pnl exists
if 'monthly_pnl' not in locals():
    monthly_pnl = daily_pnl_for_plots.resample('M')[
        'PNL_Daily'].sum() if 'daily_pnl_for_plots' in locals() else pd.Series()

if not monthly_pnl.empty:
    print(f"{'Monthly P&L Total:':<30} ${monthly_pnl.sum():,.2f}")
    print(
        f"{'Monthly Win Rate:':<30} {(monthly_pnl > 0).sum() / len(monthly_pnl) * 100:.1f}% ({monthly_pnl[monthly_pnl > 0].count()}/{len(monthly_pnl)})")
    print(f"{'Best Month:':<30} ${monthly_pnl.max():,.2f}")
    print(f"{'Average Month:':<30} ${monthly_pnl.mean():,.2f}")
    print(f"{'Worst Month:':<30} ${monthly_pnl.min():,.2f}")
else:
    print("Monthly P&L data not available")

# Yearly stats
if 'yearly_pnl' not in locals():
    yearly_pnl = daily_pnl_for_plots.resample('Y')[
        'PNL_Daily'].sum() if 'daily_pnl_for_plots' in locals() else pd.Series()

if not yearly_pnl.empty:
    print(f"{'Yearly P&L Total:':<30} ${yearly_pnl.sum():,.2f}")
    print(
        f"{'Yearly Win Rate:':<30} {(yearly_pnl > 0).sum() / len(yearly_pnl) * 100:.1f}% ({yearly_pnl[yearly_pnl > 0].count()}/{len(yearly_pnl)})")
else:
    print("Yearly P&L data not available")

# Portfolio stats
if not portfolio_pnl.empty:
    # Calculate daily portfolio P&L if not already done
    if 'portfolio_daily_pnl' not in locals():
        portfolio_daily_pnl = portfolio_pnl.diff().fillna(portfolio_pnl.iloc[0])

    portfolio_monthly_pnl = portfolio_daily_pnl.resample('M').sum()
    portfolio_yearly_pnl = portfolio_daily_pnl.resample('Y').sum()

    print("\n" + "-" * 60)
    print("PORTFOLIO (ALL ACCOUNTS) PERFORMANCE")
    print("-" * 60)

    if not portfolio_monthly_pnl.empty:
        print(f"{'Portfolio Monthly P&L Total:':<30} ${portfolio_monthly_pnl.sum():,.2f}")
        print(
            f"{'Portfolio Monthly Win Rate:':<30} {(portfolio_monthly_pnl > 0).sum() / len(portfolio_monthly_pnl) * 100:.1f}% ({portfolio_monthly_pnl[portfolio_monthly_pnl > 0].count()}/{len(portfolio_monthly_pnl)})")
        print(f"{'Portfolio Best Month:':<30} ${portfolio_monthly_pnl.max():,.2f}")
        print(f"{'Portfolio Average Month:':<30} ${portfolio_monthly_pnl.mean():,.2f}")
        print(f"{'Portfolio Worst Month:':<30} ${portfolio_monthly_pnl.min():,.2f}")

    if not portfolio_yearly_pnl.empty:
        print(f"{'Portfolio Yearly P&L Total:':<30} ${portfolio_yearly_pnl.sum():,.2f}")
        print(
            f"{'Portfolio Yearly Win Rate:':<30} {(portfolio_yearly_pnl > 0).sum() / len(portfolio_yearly_pnl) * 100:.1f}% ({portfolio_yearly_pnl[portfolio_yearly_pnl > 0].count()}/{len(portfolio_yearly_pnl)})")

print("\n" + "=" * 60)
print("\nIMPORTANT NOTE: This simulation assumes MAE occurs before MFE in each trade.")
print("This is the most conservative assumption. In reality, the sequence varies,")
print("so actual survival rates may be higher than simulated.")
print(f"\nTotal Capital Deployed across {number_accounts_started} accounts: ${total_capital_deployed:,.2f}")

try:
    plt.show()
except KeyboardInterrupt:
    print("\nScript stopped by user.")