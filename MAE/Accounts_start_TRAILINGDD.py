import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.ticker as ticker
from datetime import timedelta

# ========================================================================================
#  CONFIG
# ========================================================================================
pd.set_option('display.min_rows', 1000)
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_categories', 10)

CSV_PATH = "Merged_GG_RG_optimized_2010_2026.csv"  # Path to your CSV file with trade data

# --- Drawdown settings ---

MAX_DRAWDOWN = 1500
START_CAPITAL = MAX_DRAWDOWN  # For prop firm style, we set max drawdown equal to starting capital (100% loss = blowout)
equity_dd_freeze_trigger = START_CAPITAL + MAX_DRAWDOWN + 100
frozen_dd_floor = START_CAPITAL + 100

# --- Date range filter ---
START_DATE = "2020-01-01"
END_DATE = None

# ==================================================================
# --- Simulation Mode ---
# Live trailing: floor trails highest intraday balance
# ==================================================================
USE_TRAILING_DD = True

# ==================================================================
# --- New account start triggers ---
# ==================================================================
MAX_ACCOUNTS = 20
USE_TIME_TRIGGER = True
TIME_TRIGGER_DAYS = 30
USE_PROFIT_TRIGGER = False
START_IF_PROFIT_THRESHOLD = 1000
USE_DD_TRIGGER = False
START_IF_DD_THRESHOLD = 400
RECOVERY_LEVEL = 0
MIN_DAYS_BETWEEN_STARTS = 1

# ==================================================================
#  PLOTS SWITCHES
# ==================================================================
# Line plots
UNIFIED_EQUITY_AND_DD_PLOTS_3 = True
STARTED_ACCOUNTS_PNL_PLOT = True
PORTFOLIO_TOTAL_PNL_PLOT = True
NUMBER_OF_ACTIVE_ACCOUNTS_OVER_TIME_PLOT = False
# Bar plots
SHOW_SINGLE_ACCOUNT_DAILY_PNL_PLOT = True
SHOW_SINGLE_ACCOUNT_MONTHLY_PNL_PLOT = True
SHOW_SINGLE_ACCOUNT_YEARLY_PNL_PLOT = True
SHOW_PORTFOLIO_MONTHLY_PNL_PLOT = True
SHOW_PORTFOLIO_YEARLY_PNL_PLOT = True

# ==================================================================
#  SIMULATION ASSUMPTIONS
# ==================================================================
"""
KEY ASSUMPTIONS FOR PROP STYLE SIMULATION:

1. Intra-trade sequence: We assume MAE (worst point) occurs BEFORE MFE (best point)
   This is the most conservative assumption for blowout testing.
   In reality, the sequence is unknown - actual results may have more survivors.

2. Capital treatment: Blown accounts have their equity removed entirely from portfolio.
   This models prop firm capital allocation where blown capital is lost.

3. LIVE TRAILING LOGIC:
   - Floor = highest_intraday_balance_ever - TRAILING_DD
   - Updates in real-time on every new high (including MFE during open trades)
   - Freeze trigger: once intraday peak >= DD_FREEZE_TRIGGER, floor freezes permanently
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

    numeric_columns = ["PNL", "MAE", "MFE"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df["Entry_time"] = pd.to_datetime(df["Entry_time"])
    df["Exit_time"] = pd.to_datetime(df["Exit_time"])

    if start_date:
        df = df[df["Exit_time"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Exit_time"] <= pd.to_datetime(end_date)]

    df = df.sort_values("Entry_time").reset_index(drop=True)
    return df


def compute_prop_style_drawdown(df):
    """
    Compute prop firm style drawdown using MAE/MFE data.
    Creates a floating equity curve for the single-account DD plots.
    """
    rows = []
    equity = 0.0

    for _, r in df.sort_values("Entry_time").iterrows():
        rows.append({"time": r["Entry_time"], "equity": equity, "event": "entry"})
        rows.append({"time": r["Entry_time"], "equity": equity + r["MAE"], "event": "mae"})
        rows.append({"time": r["Entry_time"], "equity": equity + r["MFE"], "event": "mfe"})
        equity += r["PNL"]
        rows.append({"time": r["Exit_time"], "equity": equity, "event": "exit"})

    equity_curve = pd.DataFrame(rows).sort_values("time")
    equity_curve["time"] = pd.to_datetime(equity_curve["time"])

    equity_peak = 0.0
    worst_dd = 0.0
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
    daily_data["DD_Floating"] = daily_data["Equity_Low"] - daily_data["Equity_Peak"]
    daily_data["Closed_Peak"] = daily_data["Equity"].cummax()
    daily_data["DD_Closed"] = daily_data["Equity"] - daily_data["Closed_Peak"]

    return daily_data, dd_curve


def create_trade_events_with_priority(df):
    """
    Convert each trade into events with proper priority ordering.

    IMPORTANT ASSUMPTION: MAE occurs before MFE within each trade.
    This is the most conservative assumption for blowout testing.

    Priority ordering per trade:
    - 0: Entry
    - 1: MAE (worst point first — conservative)
    - 2: MFE (best point second)
    - 3: Exit
    """
    events = []
    cumulative_pnl = 0

    for trade_idx, trade in df.iterrows():
        pre_trade_equity = cumulative_pnl

        events.append({
            'time': trade['Entry_time'],
            'trade_idx': trade_idx,
            'event_type': 'entry',
            'priority': 0,
            'pre_trade_equity': pre_trade_equity,
            'mae': trade['MAE'],
            'mfe': trade['MFE'],
            'pnl': trade['PNL'],
            'equity_change': 0
        })

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
            'temp_equity': pre_trade_equity + trade['MAE']
        })

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
            'temp_equity': pre_trade_equity + trade['MFE']
        })

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
            'new_equity': cumulative_pnl
        })

    events_df = pd.DataFrame(events)
    events_df = events_df.sort_values(['time', 'priority']).reset_index(drop=True)
    return events_df


def simulate_accounts_with_prop_dd_optimized(events_df, start_capital, max_accounts):
    """
    Simulates multiple accounts using live trailing DD:

    TRAILING DD MODE:
      - DD floor = peak_intraday_equity - TRAILING_DD
      - Peak updates in real-time on MFE and profitable exits
      - Once peak >= DD_FREEZE_TRIGGER, floor freezes permanently at FROZEN_DD_FLOOR
    """
    accounts = []

    event_times   = events_df['time'].values
    event_types   = events_df['event_type'].values
    event_mae     = events_df['mae'].values
    event_mfe     = events_df['mfe'].values
    event_pnl     = events_df['pnl'].values

    total_events = len(events_df)

    last_start_date = pd.Timestamp(event_times[0])
    waiting_for_recovery = False

    portfolio_pnl_history = []
    num_alive_history = []
    portfolio_times = []
    account_history_points = []

    def make_account(account_id, start_idx, start_date):
        return {
            'id': account_id,
            'start_idx': start_idx,
            'start_date': start_date,
            'equity': start_capital,
            'pnl': 0,
            'peak': start_capital,
            'freeze_triggered': False,
            'alive': True,
            'current_trade_start_equity': None,
            'last_event_idx': -1,
        }

    accounts.append(make_account(1, 0, pd.Timestamp(event_times[0])))

    for event_idx in range(total_events):
        current_time = pd.Timestamp(event_times[event_idx])
        event_type   = event_types[event_idx]

        active_accounts = [
            acc for acc in accounts
            if acc['alive'] and acc['start_idx'] <= event_idx
        ]

        for acc in active_accounts:
            if acc['last_event_idx'] >= event_idx:
                continue
            acc['last_event_idx'] = event_idx

            # ----------------------------------------------------------------
            # ENTRY
            # ----------------------------------------------------------------
            if event_type == 'entry':
                acc['current_trade_start_equity'] = acc['equity']

            # ----------------------------------------------------------------
            # MAE — check blowout
            # ----------------------------------------------------------------
            elif event_type == 'mae':
                if acc['current_trade_start_equity'] is None or not acc['alive']:
                    continue

                temp_equity = acc['current_trade_start_equity'] + event_mae[event_idx]
                floor = (
                    frozen_dd_floor
                    if acc['freeze_triggered']
                    else acc['peak'] - MAX_DRAWDOWN
                )
                if temp_equity <= floor:
                    acc['alive'] = False
                    acc['equity'] = temp_equity
                    acc['pnl'] = temp_equity - start_capital
                    print(f"Account {acc['id']} BLOWN intraday (Trailing DD) on {current_time} | "
                          f"temp_equity={temp_equity:.2f} floor={floor:.2f}")
                    account_history_points.append({
                        'time': current_time, 'account_id': acc['id'],
                        'equity': temp_equity, 'pnl': acc['pnl'], 'event': 'blowout_mae'
                    })

            # ----------------------------------------------------------------
            # MFE — update trailing peak
            # ----------------------------------------------------------------
            elif event_type == 'mfe':
                if acc['current_trade_start_equity'] is None or not acc['alive']:
                    continue

                temp_equity = acc['current_trade_start_equity'] + event_mfe[event_idx]
                if temp_equity > acc['peak']:
                    acc['peak'] = temp_equity
                    if acc['peak'] >= equity_dd_freeze_trigger:
                        acc['freeze_triggered'] = True

            # ----------------------------------------------------------------
            # EXIT — update equity and check blowout
            # ----------------------------------------------------------------
            elif event_type == 'exit':
                if acc['current_trade_start_equity'] is None or not acc['alive']:
                    continue

                new_equity = acc['current_trade_start_equity'] + event_pnl[event_idx]
                acc['equity'] = new_equity
                acc['pnl']    = new_equity - start_capital
                acc['current_trade_start_equity'] = None

                if acc['equity'] > acc['peak']:
                    acc['peak'] = acc['equity']
                    if acc['peak'] >= equity_dd_freeze_trigger:
                        acc['freeze_triggered'] = True

                floor = (
                    frozen_dd_floor
                    if acc['freeze_triggered']
                    else acc['peak'] - MAX_DRAWDOWN
                )
                if acc['equity'] <= floor:
                    acc['alive'] = False
                    print(f"Account {acc['id']} BLOWN at exit (Trailing DD) on {current_time} | "
                          f"equity={acc['equity']:.2f} floor={floor:.2f}")
                    account_history_points.append({
                        'time': current_time, 'account_id': acc['id'],
                        'equity': acc['equity'], 'pnl': acc['pnl'], 'event': 'blowout_exit'
                    })
                    continue

                account_history_points.append({
                    'time': current_time, 'account_id': acc['id'],
                    'equity': acc['equity'], 'pnl': acc['pnl'], 'event': 'exit'
                })

        # ----------------------------------------------------------------
        # Portfolio snapshot
        # ----------------------------------------------------------------
        if event_type in ['exit', 'mae']:
            total_pnl  = sum(acc['pnl'] for acc in accounts if acc['alive'])
            alive_count = sum(1 for acc in accounts if acc['alive'])
            portfolio_pnl_history.append(total_pnl)
            num_alive_history.append(alive_count)
            portfolio_times.append(current_time)

        # ----------------------------------------------------------------
        # Start new accounts
        # ----------------------------------------------------------------
        if len(accounts) < max_accounts:
            alive_accounts = [acc for acc in accounts if acc['alive']]
            if alive_accounts:
                current_dd = min(acc['equity'] - acc['peak'] for acc in alive_accounts)
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
                trigger_dd     = False
                trigger_profit = False
                trigger_time   = False

                if USE_DD_TRIGGER and START_IF_DD_THRESHOLD:
                    if current_dd <= -START_IF_DD_THRESHOLD:
                        trigger_dd = True
                        started_due_to_dd = True

                if USE_PROFIT_TRIGGER and START_IF_PROFIT_THRESHOLD and alive_accounts:
                    last_alive = alive_accounts[-1]
                    if last_alive['equity'] - start_capital >= START_IF_PROFIT_THRESHOLD:
                        trigger_profit = True

                if USE_TIME_TRIGGER:
                    next_start_time = last_start_date + pd.Timedelta(days=TIME_TRIGGER_DAYS)
                    if current_time >= next_start_time:
                        trigger_time = True

                if trigger_dd or trigger_profit or trigger_time:
                    can_start = True

            if can_start:
                time_diff = current_time - last_start_date
                days_since_last = time_diff.total_seconds() / (24 * 3600)
                if days_since_last >= MIN_DAYS_BETWEEN_STARTS:
                    scheduled_start = last_start_date + pd.Timedelta(days=TIME_TRIGGER_DAYS)
                    new_acc = make_account(len(accounts) + 1, event_idx, scheduled_start)
                    accounts.append(new_acc)
                    print(f"Started Account {len(accounts)} scheduled for {scheduled_start} (first event {current_time})")
                    last_start_date = scheduled_start
                    waiting_for_recovery = USE_DD_TRIGGER and started_due_to_dd

    print(f"\nSimulation complete. Processed {total_events} events, {len(accounts)} accounts.")

    if portfolio_times:
        portfolio_pnl_series = pd.Series(portfolio_pnl_history, index=portfolio_times, name='portfolio_pnl')
        portfolio_pnl_daily  = portfolio_pnl_series.resample('D').last().ffill()

        num_alive_series = pd.Series(num_alive_history, index=portfolio_times, name='num_alive')
        num_alive_daily  = num_alive_series.resample('D').last().ffill()
    else:
        portfolio_pnl_daily = pd.Series()
        num_alive_daily     = pd.Series()

    if account_history_points:
        history_df = pd.DataFrame(account_history_points)
        account_pnl = history_df.pivot_table(
            index='time', columns='account_id', values='pnl', aggfunc='last'
        )
        account_pnl.columns = [f'acc_{col}_pnl' for col in account_pnl.columns]
        account_pnl = account_pnl.resample('D').last().ffill()
    else:
        account_pnl = pd.DataFrame()

    return portfolio_pnl_daily, account_pnl, num_alive_daily, accounts


def simulate_accounts_closed_dd(pl_series, start_capital, max_accounts):
    """Original closed-equity simulation (simplified, for reference)."""
    dates    = pl_series.index
    accounts = []

    accounts.append({
        'id': 1, 'start_idx': 0, 'start_date': dates[0],
        'equity': start_capital, 'pnl': 0,
        'rolling_max': start_capital, 'alive': True
    })

    last_start_date = dates[0]
    portfolio_pnl   = []
    num_alive       = []
    account_pnls    = []

    for i_date, date in enumerate(dates):
        for acc in accounts:
            if acc['alive'] and i_date >= acc['start_idx']:
                acc['equity']      += pl_series.iloc[i_date]
                acc['pnl']          = acc['equity'] - start_capital
                acc['rolling_max']  = max(acc['rolling_max'], acc['equity'])

                floor = (
                    frozen_dd_floor
                    if acc['rolling_max'] >= equity_dd_freeze_trigger
                    else acc['rolling_max'] - MAX_DRAWDOWN
                )
                if acc['equity'] <= floor:
                    acc['alive'] = False

        total_pnl = sum(acc['pnl'] for acc in accounts if acc['alive'])
        portfolio_pnl.append(total_pnl)
        num_alive.append(sum(1 for acc in accounts if acc['alive']))

        row = {
            f'acc_{acc["id"]}_pnl': acc['pnl'] if acc['start_idx'] <= i_date else np.nan
            for acc in accounts
        }
        row['time'] = date
        account_pnls.append(row)

        if len(accounts) < max_accounts and USE_TIME_TRIGGER:
            if (date - last_start_date).days >= TIME_TRIGGER_DAYS:
                accounts.append({
                    'id': len(accounts) + 1, 'start_idx': i_date, 'start_date': date,
                    'equity': start_capital, 'pnl': 0,
                    'rolling_max': start_capital, 'alive': True
                })
                last_start_date = date

    portfolio_pnl_series = pd.Series(portfolio_pnl, index=dates, name='portfolio_pnl')
    accounts_df          = pd.DataFrame(account_pnls).set_index('time')
    num_alive_series     = pd.Series(num_alive, index=dates, name='num_alive')

    return portfolio_pnl_series, accounts_df, num_alive_series, accounts


def print_config():
    print("=== Configuration ===")
    print(f"CSV_PATH:          {CSV_PATH}")
    print(f"Mode:              Trailing DD")
    print(f"START_CAPITAL:     {START_CAPITAL}")
    print(f"TRAILING_DD:       {MAX_DRAWDOWN}")
    print(f"DD_FREEZE_TRIGGER: {equity_dd_freeze_trigger}")
    print(f"FROZEN_DD_FLOOR:   {frozen_dd_floor}")
    if START_DATE: print(f"START_DATE:        {START_DATE}")
    if END_DATE:   print(f"END_DATE:          {END_DATE}")
    print(f"MAX_ACCOUNTS:      {MAX_ACCOUNTS}")
    print(f"TIME_TRIGGER_DAYS: {TIME_TRIGGER_DAYS}")
    print("=====================")


# ======================
#  MAIN EXECUTION
# ======================
print_config()

df = load_and_preprocess_data(CSV_PATH, START_DATE, END_DATE)
print(f"\nLoaded {len(df)} trades from {df['Entry_time'].min()} to {df['Exit_time'].max()}")

mode = "Prop Firm — Live Trailing DD"

print("\nComputing prop-style drawdown for visualization...")
daily_data, full_dd_curve = compute_prop_style_drawdown(df)
plot_df = daily_data.copy()

daily_pnl_for_plots = daily_data[["Date", "Equity"]].copy()
daily_pnl_for_plots.rename(columns={"Equity": "PNL_Daily"}, inplace=True)
daily_pnl_for_plots["PNL_Daily"] = daily_pnl_for_plots["PNL_Daily"].diff()
if len(daily_pnl_for_plots) > 0:
    daily_pnl_for_plots.iloc[0, daily_pnl_for_plots.columns.get_loc('PNL_Daily')] = \
        daily_data["Equity"].iloc[0]
daily_pnl_for_plots.set_index('Date', inplace=True)

print("\n" + "=" * 60)
print("SIMULATION MODE:", mode)
print("=" * 60)

events_df = create_trade_events_with_priority(df)
print(f"Created {len(events_df)} events from {len(df)} trades")
print(f"Event type distribution:\n{events_df['event_type'].value_counts()}")

print(f"\nRunning simulation ({mode})...")
portfolio_pnl, acc_pnl_df, num_alive_df, accounts = simulate_accounts_with_prop_dd_optimized(
    events_df, START_CAPITAL, MAX_ACCOUNTS
)

# ============================================================
# UNIFIED EQUITY + DD PLOTS (Single Account Strategy)
# ============================================================

if UNIFIED_EQUITY_AND_DD_PLOTS_3:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(plot_df["Date"], plot_df["Equity"],      linewidth=2, label="Equity")
    axes[0].plot(plot_df["Date"], plot_df["Equity_Peak"], linewidth=1, label="Equity_Peak")
    axes[0].plot(plot_df["Date"], plot_df["Equity_Low"],  linewidth=1, label="Equity_Low")
    axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    axes[0].set_title("Equity Curve (Single Account Strategy)")
    axes[0].set_ylabel("Equity ($)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(plot_df["Date"], plot_df["DD_Closed"], linewidth=2, label="Closed DD")
    axes[1].axhline(0, linewidth=0.8)
    axes[1].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    axes[1].set_title("Closed Equity Drawdown")
    axes[1].set_ylabel("Drawdown ($)")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(plot_df["Date"], plot_df["DD_Floating"], linewidth=2, label="Floating DD")
    axes[2].axhline(0, linewidth=0.8)
    axes[2].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    axes[2].set_title("Floating Drawdown (Includes Intraday MAE/MFE)")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Drawdown ($)")
    axes[2].grid(True)
    axes[2].legend()

    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()


# ======================
# PORTFOLIO TOTAL PNL PLOT
# ======================

# --- Additional portfolio curves ---
portfolio_all_accounts = acc_pnl_df.sum(axis=1)

portfolio_alive_accounts = acc_pnl_df.copy()
for acc in accounts:
    if not acc['alive']:
        col = f'acc_{acc["id"]}_pnl'
        if col in portfolio_alive_accounts.columns:
            portfolio_alive_accounts[col] = np.nan

portfolio_alive_accounts = portfolio_alive_accounts.sum(axis=1)

portfolio_profitable_accounts = acc_pnl_df.copy()
for col in portfolio_profitable_accounts.columns:
    portfolio_profitable_accounts[col] = portfolio_profitable_accounts[col].clip(lower=0)

portfolio_profitable_accounts = portfolio_profitable_accounts.sum(axis=1)

if PORTFOLIO_TOTAL_PNL_PLOT and not portfolio_pnl.empty:
    fig_portfolio, ax_portfolio = plt.subplots(figsize=(14, 6))
    ax_portfolio.plot(
        portfolio_all_accounts.index,
        portfolio_all_accounts.values,
        linewidth=2,
        color='orange',
        label="Strategy P&L - all accounts(alive, blown, and in a loss)"
    )
    ax_portfolio.plot(
        portfolio_alive_accounts.index,
        portfolio_alive_accounts.values,
        linewidth=3,
        color='blue',
        label="Portfolio P&L - alive accounts(in profit and in a loss)"
    )
    ax_portfolio.plot(
        portfolio_profitable_accounts.index,
        portfolio_profitable_accounts.values,
        linewidth=2,
        color='green',
        label="Withdrawable P&L - profitable accounts - (only in profit)"
    )

    ax_portfolio.axhline(y=0, color='black', linewidth=0.8, alpha=0.5, linestyle='--')

    ax_portfolio.set_title(f"Portfolio P&L Comparison — {mode}")
    ax_portfolio.set_ylabel("P&L ($)")
    ax_portfolio.set_xlabel("Date")
    ax_portfolio.grid(True, alpha=0.3)
    ax_portfolio.legend()
    ax_portfolio.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax_portfolio.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_portfolio.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    plt.setp(ax_portfolio.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()

# ======================
# INDIVIDUAL ACCOUNTS P&L PLOT
# ======================

if STARTED_ACCOUNTS_PNL_PLOT and not acc_pnl_df.empty:
    fig_accounts, ax_accounts = plt.subplots(figsize=(14, 6))
    number_accounts_started = len(accounts)

    for i, col in enumerate(acc_pnl_df.columns[:number_accounts_started]):
        ax_accounts.plot(
            acc_pnl_df.index, acc_pnl_df[col],
            alpha=0.7, linewidth=1.5,
            label=f"Account {i + 1}" if i < 10 else None
        )

    ax_accounts.set_title(f"Individual Accounts P&L — {mode}")
    ax_accounts.set_ylabel("P&L ($)")
    ax_accounts.set_xlabel("Date")
    ax_accounts.yaxis.set_major_formatter(
        mticker.StrMethodFormatter('{x:,.0f}')
    )
    ax_accounts.grid(True, alpha=0.3)
    ax_accounts.axhline(y=0, color='black', linewidth=0.8, alpha=0.5, linestyle='--')

    ax_accounts.axhline(
        y=frozen_dd_floor,
        color='red', linewidth=2, linestyle='--', alpha=0.8,
        label=f'Frozen DD floor (equity={frozen_dd_floor})'
    )
    ax_accounts.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax_accounts.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax_accounts.xaxis.get_majorticklabels(), rotation=45)
    ax_accounts.legend()
    plt.tight_layout()

# ======================
# ACCOUNT STATUS OVER TIME
# ======================

if NUMBER_OF_ACTIVE_ACCOUNTS_OVER_TIME_PLOT and not num_alive_df.empty:
    fig3, ax5 = plt.subplots(1, 1, figsize=(14, 6))
    number_accounts_started = len(accounts)

    ax5.plot(num_alive_df.index, num_alive_df.values,
             color="purple", linewidth=3, label="Number of Alive Accounts")
    ax5.fill_between(num_alive_df.index, num_alive_df.values, 0,
                     color="purple", alpha=0.2)
    ax5.set_title(f"Number of Active Accounts Over Time — {mode}")
    ax5.set_ylabel("Number of Accounts")
    ax5.set_xlabel("Date")
    ax5.grid(True, alpha=0.3)

    ax5.axhline(y=number_accounts_started, color='gray', linestyle='--',
                alpha=0.5, label=f"Total Started: {number_accounts_started}")
    ax5.axhline(y=num_alive_df.iloc[-1], color='green', linestyle='--',
                alpha=0.5, label=f"Final Alive: {num_alive_df.iloc[-1]}")
    ax5.legend(loc='upper left')

    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()

# ============================================================
# CHART 0: DAILY P&L (Single Account Strategy)
# ============================================================

if SHOW_SINGLE_ACCOUNT_DAILY_PNL_PLOT:
    daily_pnl_series = daily_pnl_for_plots['PNL_Daily']

    fig_daily, ax_daily = plt.subplots(figsize=(16, 6))
    colors = ['green' if x >= 0 else 'red' for x in daily_pnl_series.values]

    ax_daily.bar(
        daily_pnl_series.index, daily_pnl_series.values,
        color=colors, alpha=0.7, edgecolor='black', linewidth=0.3
    )
    ax_daily.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)

    ax_daily.xaxis.set_major_locator(ticker.MaxNLocator(20))
    ax_daily.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.setp(ax_daily.xaxis.get_majorticklabels(), rotation=60)

    ax_daily.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    ax_daily.yaxis.set_major_locator(mticker.MultipleLocator(50))

    ax_daily.set_title("Single Account Strategy - Daily P&L", fontsize=14, fontweight='bold')
    ax_daily.set_ylabel("P&L ($)")
    ax_daily.set_xlabel("Date")
    ax_daily.grid(True, alpha=0.3, axis='y')
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

if SHOW_SINGLE_ACCOUNT_MONTHLY_PNL_PLOT:
    monthly_pnl = daily_pnl_for_plots.resample('M')['PNL_Daily'].sum()

    fig_monthly, ax_monthly = plt.subplots(figsize=(14, 6))
    colors = ['green' if x >= 0 else 'red' for x in monthly_pnl.values]
    bars   = ax_monthly.bar(monthly_pnl.index, monthly_pnl.values,
                            color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_monthly.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)

    ax_monthly.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_monthly.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax_monthly.xaxis.get_majorticklabels(), rotation=45)

    for bar in bars:
        height = bar.get_height()
        label_pos = height + (monthly_pnl.max() * 0.01) if height >= 0 else height - (monthly_pnl.max() * 0.01)
        ax_monthly.text(bar.get_x() + bar.get_width() / 2., label_pos,
                        f'${height:,.0f}', ha='center',
                        va='bottom' if height >= 0 else 'top', fontsize=8, rotation=45)

    ax_monthly.set_title("Single Account Strategy - Monthly P&L", fontsize=14, fontweight='bold')
    ax_monthly.set_ylabel("P&L ($)")
    ax_monthly.set_xlabel("Date")
    ax_monthly.grid(True, alpha=0.3, axis='y')

    total_pnl       = monthly_pnl.sum()
    positive_months = (monthly_pnl > 0).sum()
    win_rate        = positive_months / len(monthly_pnl) * 100 if len(monthly_pnl) > 0 else 0
    textstr = f'Total: ${total_pnl:,.0f} | Win Rate: {win_rate:.1f}% ({positive_months}/{len(monthly_pnl)})'
    ax_monthly.text(0.02, 0.98, textstr, transform=ax_monthly.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

else:
    monthly_pnl = daily_pnl_for_plots.resample('M')['PNL_Daily'].sum()

# ============================================================
# CHART 2: YEARLY P&L (Single Account Strategy)
# ============================================================

if SHOW_SINGLE_ACCOUNT_YEARLY_PNL_PLOT:
    yearly_pnl = daily_pnl_for_plots.resample('Y')['PNL_Daily'].sum()

    fig_yearly, ax_yearly = plt.subplots(figsize=(12, 6))
    colors = ['green' if x >= 0 else 'red' for x in yearly_pnl.values]
    bars   = ax_yearly.bar(yearly_pnl.index.year, yearly_pnl.values,
                           color=colors, alpha=0.7, edgecolor='black', linewidth=0.8, width=0.6)
    ax_yearly.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)

    for bar in bars:
        height    = bar.get_height()
        label_pos = height + (yearly_pnl.max() * 0.02) if height >= 0 else height - (yearly_pnl.max() * 0.02)
        ax_yearly.text(bar.get_x() + bar.get_width() / 2., label_pos,
                       f'${height:,.0f}', ha='center',
                       va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')

    ax_yearly.set_title("Single Account Strategy - Yearly P&L", fontsize=14, fontweight='bold')
    ax_yearly.set_ylabel("P&L ($)")
    ax_yearly.set_xlabel("Year")
    ax_yearly.grid(True, alpha=0.3, axis='y')

    total_pnl_yearly = yearly_pnl.sum()
    avg_yearly       = yearly_pnl.mean() if len(yearly_pnl) > 0 else 0
    positive_years   = (yearly_pnl > 0).sum()
    win_rate_yearly  = positive_years / len(yearly_pnl) * 100 if len(yearly_pnl) > 0 else 0
    textstr = f'Total: ${total_pnl_yearly:,.0f} | Avg: ${avg_yearly:,.0f} | Win Rate: {win_rate_yearly:.1f}% ({positive_years}/{len(yearly_pnl)})'
    ax_yearly.text(0.02, 0.98, textstr, transform=ax_yearly.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

else:
    yearly_pnl = daily_pnl_for_plots.resample('Y')['PNL_Daily'].sum()

# ============================================================
# CHART 3: PORTFOLIO MONTHLY P&L
# ============================================================

if SHOW_PORTFOLIO_MONTHLY_PNL_PLOT and not portfolio_pnl.empty:
    portfolio_daily_pnl   = portfolio_pnl.diff().fillna(portfolio_pnl.iloc[0])
    portfolio_monthly_pnl = portfolio_daily_pnl.resample('M').sum()

    fig_portfolio_monthly, ax_portfolio_monthly = plt.subplots(figsize=(14, 6))
    colors = ['green' if x >= 0 else 'red' for x in portfolio_monthly_pnl.values]
    bars   = ax_portfolio_monthly.bar(portfolio_monthly_pnl.index, portfolio_monthly_pnl.values,
                                      color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_portfolio_monthly.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)

    ax_portfolio_monthly.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_portfolio_monthly.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax_portfolio_monthly.xaxis.get_majorticklabels(), rotation=45)

    for bar in bars:
        height    = bar.get_height()
        label_pos = height + (portfolio_monthly_pnl.max() * 0.01) if height >= 0 else height - (portfolio_monthly_pnl.max() * 0.01)
        ax_portfolio_monthly.text(bar.get_x() + bar.get_width() / 2., label_pos,
                                  f'${height:,.0f}', ha='center',
                                  va='bottom' if height >= 0 else 'top', fontsize=8, rotation=45)

    ax_portfolio_monthly.set_title(f"Portfolio (All Accounts) - Monthly P&L — {mode}", fontsize=14, fontweight='bold')
    ax_portfolio_monthly.set_ylabel("P&L ($)")
    ax_portfolio_monthly.set_xlabel("Date")
    ax_portfolio_monthly.grid(True, alpha=0.3, axis='y')

    portfolio_total_pnl    = portfolio_monthly_pnl.sum()
    portfolio_pos_months   = (portfolio_monthly_pnl > 0).sum()
    portfolio_win_rate     = portfolio_pos_months / len(portfolio_monthly_pnl) * 100 if len(portfolio_monthly_pnl) > 0 else 0
    textstr = f'Total: ${portfolio_total_pnl:,.0f} | Win Rate: {portfolio_win_rate:.1f}% ({portfolio_pos_months}/{len(portfolio_monthly_pnl)})'
    ax_portfolio_monthly.text(0.02, 0.98, textstr, transform=ax_portfolio_monthly.transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
else:
    portfolio_daily_pnl = portfolio_pnl.diff().fillna(portfolio_pnl.iloc[0])
    portfolio_monthly_pnl = portfolio_daily_pnl.resample('M').sum()

# ============================================================
# CHART 4: PORTFOLIO YEARLY P&L
# ============================================================

if SHOW_PORTFOLIO_YEARLY_PNL_PLOT and not portfolio_pnl.empty:
    portfolio_yearly_pnl = portfolio_daily_pnl.resample('Y').sum()

    fig_portfolio_yearly, ax_portfolio_yearly = plt.subplots(figsize=(12, 6))
    colors = ['green' if x >= 0 else 'red' for x in portfolio_yearly_pnl.values]
    bars   = ax_portfolio_yearly.bar(portfolio_yearly_pnl.index.year, portfolio_yearly_pnl.values,
                                     color=colors, alpha=0.7, edgecolor='black', linewidth=0.8, width=0.6)
    ax_portfolio_yearly.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)

    for bar in bars:
        height    = bar.get_height()
        label_pos = height + (portfolio_yearly_pnl.max() * 0.02) if height >= 0 else height - (portfolio_yearly_pnl.max() * 0.02)
        ax_portfolio_yearly.text(bar.get_x() + bar.get_width() / 2., label_pos,
                                 f'${height:,.0f}', ha='center',
                                 va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')

    ax_portfolio_yearly.set_title(f"Portfolio (All Accounts) - Yearly P&L — {mode}", fontsize=14, fontweight='bold')
    ax_portfolio_yearly.set_ylabel("P&L ($)")
    ax_portfolio_yearly.set_xlabel("Year")
    ax_portfolio_yearly.grid(True, alpha=0.3, axis='y')

    portfolio_total_pnl_yearly = portfolio_yearly_pnl.sum()
    portfolio_avg_yearly       = portfolio_yearly_pnl.mean() if len(portfolio_yearly_pnl) > 0 else 0
    portfolio_pos_years        = (portfolio_yearly_pnl > 0).sum()
    portfolio_win_rate_yearly  = portfolio_pos_years / len(portfolio_yearly_pnl) * 100 if len(portfolio_yearly_pnl) > 0 else 0
    textstr = f'Total: ${portfolio_total_pnl_yearly:,.0f} | Avg: ${portfolio_avg_yearly:,.0f} | Win Rate: {portfolio_win_rate_yearly:.1f}% ({portfolio_pos_years}/{len(portfolio_yearly_pnl)})'
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

print("\nFINAL P&L PER ACCOUNT:")
print("-" * 60)
for acc in accounts:
    status = "ALIVE" if acc['alive'] else "BLOWN \u2B24"
    peak_pnl = acc['peak'] - START_CAPITAL
    print(f"Account {acc['id']:>2} | Status: {status:<8} | Final P&L: ${acc['pnl']:>8.2f} | Highest P&L(MFE): ${peak_pnl:>8.2f}")
print("-" * 60)

number_accounts_started = len(accounts)
number_accounts_alive   = sum(1 for acc in accounts if acc['alive'])
number_accounts_blown   = number_accounts_started - number_accounts_alive

# --- Portfolio metrics ---
strategy_total_pnl = sum(acc['pnl'] for acc in accounts)
portfolio_alive_pnl = sum(acc['pnl'] for acc in accounts if acc['alive'])
portfolio_profitable_pnl = sum(
    acc['pnl'] for acc in accounts
    if acc['alive'] and acc['pnl'] > 0
)

total_capital_deployed = START_CAPITAL * number_accounts_started

print(f"{'Simulation Mode:':<35} {mode}")
print()

print("PNL OVERVIEW")
print("-" * 60)
print(f"{'Portfolio P&L (profitable survivors):':<40} ${portfolio_profitable_pnl:,.2f}")
print(f"{'Strategy Total P&L (all accounts):':<40} ${strategy_total_pnl:,.2f}")
print(f"{'Portfolio P&L (alive accounts):':<40} ${portfolio_alive_pnl:,.2f}")
print()

print("ACCOUNT STATISTICS")
print("-" * 60)
print(f"{'Accounts Started:':<35} {number_accounts_started}")
print(f"{'Accounts Alive:':<35} {number_accounts_alive}")
print(f"{'Accounts Blown:':<35} {number_accounts_blown}")
print(f"{'Survival Rate:':<35} {(number_accounts_alive / number_accounts_started * 100):.1f}%")

print()

print("CAPITAL METRICS")
print("-" * 60)
print(f"{'Total Capital Deployed:':<35} ${total_capital_deployed:,.2f}")
try:
    print(f"{'Return on Capital (alive PnL):':<35} {(portfolio_profitable_pnl / total_capital_deployed * 100):.1f}%")
except ZeroDivisionError:
    print(f"{'Return on Capital (alive PnL):':<35} N/A (no capital deployed or blown)")

freeze_count = sum(1 for acc in accounts if acc['freeze_triggered'])

print("\nBLOWOUT ANALYSIS")
print("-" * 60)
print(f"{'Accounts that hit freeze trigger:':<35} {freeze_count}")
print(f"{'Total Blown Accounts:':<35} {number_accounts_blown}")

print("\n" + "-" * 60)
print("SINGLE ACCOUNT STRATEGY PERFORMANCE")
print("-" * 60)

if not monthly_pnl.empty:
    print(f"{'Monthly P&L Total:':<35} ${monthly_pnl.sum():,.2f}")
    print(f"{'Monthly Win Rate:':<35} {(monthly_pnl > 0).sum() / len(monthly_pnl) * 100:.1f}%"
          f" ({(monthly_pnl > 0).sum()}/{len(monthly_pnl)})")
    print(f"{'Best Month:':<35} ${monthly_pnl.max():,.2f}")
    print(f"{'Average Month:':<35} ${monthly_pnl.mean():,.2f}")
    print(f"{'Worst Month:':<35} ${monthly_pnl.min():,.2f}")

if not yearly_pnl.empty:
    print(f"{'Yearly P&L Total:':<35} ${yearly_pnl.sum():,.2f}")
    print(f"{'Yearly Win Rate:':<35} {(yearly_pnl > 0).sum() / len(yearly_pnl) * 100:.1f}%"
          f" ({(yearly_pnl > 0).sum()}/{len(yearly_pnl)})")

if not portfolio_pnl.empty:
    portfolio_monthly_pnl = portfolio_daily_pnl.resample('M').sum()
    portfolio_yearly_pnl  = portfolio_daily_pnl.resample('Y').sum()

    print("\n" + "-" * 60)
    print("PORTFOLIO (ALL ACCOUNTS) PERFORMANCE")
    print("-" * 60)

    if not portfolio_monthly_pnl.empty:
        print(f"{'Portfolio Monthly P&L Total:':<35} ${portfolio_monthly_pnl.sum():,.2f}")
        print(f"{'Portfolio Monthly Win Rate:':<35} {(portfolio_monthly_pnl > 0).sum() / len(portfolio_monthly_pnl) * 100:.1f}%"
              f" ({(portfolio_monthly_pnl > 0).sum()}/{len(portfolio_monthly_pnl)})")
        print(f"{'Portfolio Best Month:':<35} ${portfolio_monthly_pnl.max():,.2f}")
        print(f"{'Portfolio Average Month:':<35} ${portfolio_monthly_pnl.mean():,.2f}")
        print(f"{'Portfolio Worst Month:':<35} ${portfolio_monthly_pnl.min():,.2f}")

    if not portfolio_yearly_pnl.empty:
        print(f"{'Portfolio Yearly P&L Total:':<35} ${portfolio_yearly_pnl.sum():,.2f}")
        print(f"{'Portfolio Yearly Win Rate:':<35} {(portfolio_yearly_pnl > 0).sum() / len(portfolio_yearly_pnl) * 100:.1f}%"
              f" ({(portfolio_yearly_pnl > 0).sum()}/{len(portfolio_yearly_pnl)})")

print("\n" + "=" * 60)
print("\nNOTE: Simulation assumes MAE occurs before MFE in each trade (conservative).")
print(f"Total Capital Deployed across {number_accounts_started} accounts: ${total_capital_deployed:,.2f}")

try:
    plt.show()
except KeyboardInterrupt:
    print("\nScript stopped by user.")