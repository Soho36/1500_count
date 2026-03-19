import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import timedelta

# ========================================================================================
#  CONFIG
# ========================================================================================
pd.set_option('display.min_rows', 1000)
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_categories', 10)

CSV_PATH = "databento_premarket_eval_10_1.16.csv"  # Path to your CSV file with trade data

# --- Drawdown settings ---

MAX_DRAWDOWN = 2000
START_CAPITAL = MAX_DRAWDOWN
equity_dd_freeze_trigger = START_CAPITAL + MAX_DRAWDOWN + 100
frozen_dd_floor = START_CAPITAL + 100

# --- Profit target ---
PROFIT_TARGET = 3000

# --- Date range filter ---
START_DATE = None
END_DATE = None

# ==================================================================
# --- New account start triggers ---
# ==================================================================
MAX_ACCOUNTS = 100
USE_TIME_TRIGGER = True
TIME_TRIGGER_DAYS = 10
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
MONTHLY_OUTCOMES_BAR_PLOT = True   # NEW: monthly PASSED / PASSEDBLOWN / BLOWN bar chart

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

3. EOD THRESHOLD LOGIC (per prop firm docs):
   - At END of each trading day, the closing account balance is recorded
   - The DD floor for the NEXT session = closing_balance - TRAILING_DD
   - This floor is FIXED for the entire next session (does NOT trail intraday)
   - But it IS enforced in real-time: if equity touches the floor intraday → blown
   - Freeze trigger: once peak closing balance >= DD_FREEZE_TRIGGER, floor freezes at FROZEN_DD_FLOOR
   - Note: freeze is evaluated at close, so it applies from the NEXT session onward

4. PROFIT TARGET:
   - When an account's closed P&L reaches or exceeds PROFIT_TARGET, it is marked as PASSED
   - The account stops trading immediately (eval complete)
   - Checked at every trade exit
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
    accounts = []

    event_times = events_df['time'].values
    event_types = events_df['event_type'].values
    event_mae   = events_df['mae'].values
    event_pnl   = events_df['pnl'].values

    total_events = len(events_df)

    last_start_date      = pd.Timestamp(event_times[0])
    waiting_for_recovery = False

    portfolio_pnl_history  = []
    num_alive_history      = []
    portfolio_times        = []
    account_history_points = []

    def make_account(account_id, start_idx, start_date):
        return {
            'id':           account_id,
            'start_idx':    start_idx,
            'start_date':   start_date,
            'equity':       start_capital,
            'pnl':          0,
            'freeze_triggered': False,
            'passed':       False,          # True when profit target hit
            'pass_date':    None,           # timestamp when target was reached
            'pass_pnl':     None,           # exact P&L at the moment of passing
            'blow_date':    None,           # timestamp when account was blown
            # --- EOD DD state ---
            'eod_dd_floor':       start_capital - MAX_DRAWDOWN,
            'eod_peak_closing':   start_capital,
            'eod_closing_equity': start_capital,
            'last_eod_date':      None,
            'peak_closed_pnl':    0.0,
            # --- Shared state ---
            'alive':                    True,
            'current_trade_start_equity': None,
            'last_event_idx':           -1,
        }

    accounts.append(make_account(1, 0, pd.Timestamp(event_times[0])))

    for event_idx in range(total_events):
        current_time = pd.Timestamp(event_times[event_idx])
        current_day  = current_time.date()
        event_type   = event_types[event_idx]

        # ----------------------------------------------------------------
        # EOD FLOOR UPDATE
        # ----------------------------------------------------------------
        if event_type == 'entry':
            for acc in accounts:
                if not acc['alive']:
                    continue
                if acc['last_eod_date'] is not None and acc['last_eod_date'] < current_day:
                    if not acc['freeze_triggered']:
                        closing_eq = acc['eod_closing_equity']
                        new_peak = max(acc['eod_peak_closing'], closing_eq)
                        acc['eod_peak_closing'] = new_peak
                        if acc['eod_peak_closing'] >= equity_dd_freeze_trigger:
                            acc['freeze_triggered'] = True
                            acc['eod_dd_floor']     = frozen_dd_floor
                            print(f"Account {acc['id']} FREEZE triggered at session open "
                                  f"on {current_day} | "
                                  f"peak_closing={acc['eod_peak_closing']:.2f}")
                        else:
                            acc['eod_dd_floor'] = max(
                                acc['eod_dd_floor'],
                                acc['eod_peak_closing'] - MAX_DRAWDOWN
                            )
                    acc['last_eod_date'] = current_day

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
            # MAE
            # ----------------------------------------------------------------
            elif event_type == 'mae':
                if acc['current_trade_start_equity'] is None or not acc['alive']:
                    continue

                temp_equity = acc['current_trade_start_equity'] + event_mae[event_idx]
                floor = acc['eod_dd_floor']
                if temp_equity <= floor:
                    acc['alive']      = False
                    acc['equity']     = temp_equity
                    acc['pnl']        = temp_equity - start_capital
                    acc['blow_date']  = current_time
                    print(f"Account {acc['id']} BLOWN intraday MAE (EOD floor) on {current_time} | "
                          f"temp_equity={temp_equity:.2f} floor={floor:.2f}")
                    account_history_points.append({
                        'time': current_time, 'account_id': acc['id'],
                        'equity': temp_equity, 'pnl': acc['pnl'], 'event': 'blowout_mae'
                    })

            # ----------------------------------------------------------------
            # MFE
            # ----------------------------------------------------------------
            elif event_type == 'mfe':
                pass

            # ----------------------------------------------------------------
            # EXIT — settle trade, check floor, check profit target
            # ----------------------------------------------------------------
            elif event_type == 'exit':
                if acc['current_trade_start_equity'] is None or not acc['alive']:
                    continue

                new_equity = acc['current_trade_start_equity'] + event_pnl[event_idx]
                acc['equity'] = new_equity
                acc['pnl']    = new_equity - start_capital
                acc['current_trade_start_equity'] = None

                # --- Blowout check ---
                floor = acc['eod_dd_floor']
                if acc['equity'] <= floor:
                    acc['alive']     = False
                    acc['blow_date'] = current_time
                    print(f"Account {acc['id']} BLOWN at exit (EOD floor) on {current_time} | "
                          f"equity={acc['equity']:.2f} floor={floor:.2f}")
                    account_history_points.append({
                        'time': current_time, 'account_id': acc['id'],
                        'equity': acc['equity'], 'pnl': acc['pnl'], 'event': 'blowout_exit'
                    })
                    continue

                # --- Profit target check ---
                if acc['pnl'] >= PROFIT_TARGET:
                    acc['alive']     = False   # stop trading — eval passed
                    acc['passed']    = True
                    acc['pass_date'] = current_time
                    acc['pass_pnl']  = acc['pnl']
                    print(f"Account {acc['id']} PASSED profit target on {current_time} | "
                          f"P&L=${acc['pnl']:.2f} (target=${PROFIT_TARGET:.2f})")
                    account_history_points.append({
                        'time': current_time, 'account_id': acc['id'],
                        'equity': acc['equity'], 'pnl': acc['pnl'], 'event': 'passed'
                    })
                    continue

                # --- Normal exit recording ---
                acc['eod_closing_equity'] = acc['equity']
                acc['last_eod_date']      = current_day
                acc['peak_closed_pnl']    = max(acc['peak_closed_pnl'], acc['pnl'])

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
    print(f"Mode:              EOD Threshold")
    print(f"START_CAPITAL:     {START_CAPITAL}")
    print(f"TRAILING_DD:       {MAX_DRAWDOWN}")
    print(f"PROFIT_TARGET:     {PROFIT_TARGET}")
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

mode = "Prop Firm — EOD Threshold"

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

print("\nEOD THRESHOLD RULES:")
print("1. DD floor is calculated ONCE per day at market close: close_equity - MAX_DRAWDOWN")
print("2. Floor is FIXED for the entire next session (does not trail intraday)")
print("3. Floor is still enforced in real-time: MAE touching floor = blown intraday")
print(f"4. Freeze trigger: once peak closing equity >= {equity_dd_freeze_trigger}, floor frozen at {frozen_dd_floor}")
print(f"5. Profit target: ${PROFIT_TARGET:,.0f} — account stops trading when reached")
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
    ax_accounts.axhline(
        y=PROFIT_TARGET,
        color='green', linewidth=2, linestyle='--', alpha=0.8,
        label=f'Profit Target (P&L=${PROFIT_TARGET:,.0f})'
    )
    ax_accounts.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax_accounts.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax_accounts.xaxis.get_majorticklabels(), rotation=45)
    ax_accounts.legend()
    plt.tight_layout()

# ============================================================
# MONTHLY OUTCOMES BAR PLOT
# ============================================================

if MONTHLY_OUTCOMES_BAR_PLOT:
    # Build a per-account outcome record using the end date of each account
    outcome_records = []
    for acc in accounts:
        if acc.get('passed', False):
            days = (acc['pass_date'] - acc['start_date']).days
            outcome = 'PASSEDBLOWN' if days > 30 else 'PASSED'
            end_date = acc['pass_date']
        elif acc['blow_date'] is not None:
            outcome = 'BLOWN'
            end_date = acc['blow_date']
        else:
            # Still active — skip from this chart
            continue
        outcome_records.append({'month': end_date.to_period('M'), 'outcome': outcome})

    if outcome_records:
        outcomes_df = pd.DataFrame(outcome_records)
        monthly_outcomes = (
            outcomes_df
            .groupby(['month', 'outcome'])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

        # Ensure all three columns exist even if a category had zero in all months
        for col in ['PASSED', 'PASSEDBLOWN', 'BLOWN']:
            if col not in monthly_outcomes.columns:
                monthly_outcomes[col] = 0
        monthly_outcomes = monthly_outcomes[['PASSED', 'PASSEDBLOWN', 'BLOWN']]

        month_labels = [str(m) for m in monthly_outcomes.index]
        x = np.arange(len(month_labels))
        bar_width = 0.25

        fig_mo, ax_mo = plt.subplots(figsize=(max(10, len(month_labels) * 1.1), 6))

        bars_passed = ax_mo.bar(
            x - bar_width, monthly_outcomes['PASSED'],
            width=bar_width, color='#2ecc71', edgecolor='white', linewidth=0.6,
            label='PASSED ✓'
        )
        bars_pb = ax_mo.bar(
            x, monthly_outcomes['PASSEDBLOWN'],
            width=bar_width, color='#f39c12', edgecolor='white', linewidth=0.6,
            label=f'PASSEDBLOWN ✗ (>{30}d)'
        )
        bars_blown = ax_mo.bar(
            x + bar_width, monthly_outcomes['BLOWN'],
            width=bar_width, color='#e74c3c', edgecolor='white', linewidth=0.6,
            label='BLOWN ✗'
        )

        # Value labels on top of each bar
        for bars in [bars_passed, bars_pb, bars_blown]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax_mo.text(
                        bar.get_x() + bar.get_width() / 2, h + 0.05,
                        str(int(h)), ha='center', va='bottom', fontsize=9, fontweight='bold'
                    )

        ax_mo.set_xticks(x)
        ax_mo.set_xticklabels(month_labels, rotation=45, ha='right')
        ax_mo.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax_mo.set_title(
            f"Monthly Account Outcomes — {mode}\n"
            f"(Profit Target: ${PROFIT_TARGET:,.0f}  |  Max DD: ${MAX_DRAWDOWN:,.0f}  |  Time Limit: {TIME_TRIGGER_DAYS}d)",
            fontsize=12
        )
        ax_mo.set_ylabel("Number of Accounts")
        ax_mo.set_xlabel("Month (outcome date)")
        ax_mo.grid(True, alpha=0.3, axis='y')
        ax_mo.legend()
        plt.tight_layout()

monthly_pnl = daily_pnl_for_plots.resample('M')['PNL_Daily'].sum()
yearly_pnl = daily_pnl_for_plots.resample('Y')['PNL_Daily'].sum()

# ======================
#  STATISTICS
# ======================
print("\n" + "=" * 60)
print("SIMULATION RESULTS")
print("=" * 60)

number_accounts_started     = len(accounts)
number_accounts_passed      = sum(1 for acc in accounts if acc.get('passed', False) and (acc['pass_date'] - acc['start_date']).days <= 30)
number_accounts_passedblown = sum(1 for acc in accounts if acc.get('passed', False) and (acc['pass_date'] - acc['start_date']).days > 30)
number_accounts_blown       = sum(1 for acc in accounts if not acc['alive'] and not acc.get('passed', False))
number_accounts_alive       = sum(1 for acc in accounts if acc['alive'])

print(f"\nEVALUATION ACCOUNTS SUMMARY  (Profit Target: ${PROFIT_TARGET:,.0f})")
print("-" * 60)
print(f"{'Accounts Started:':<35} {number_accounts_started}")
print(f"{'Accounts PASSED (target hit):':<35} {number_accounts_passed}  "
      f"({number_accounts_passed / number_accounts_started * 100:.1f}%)")
print(f"{'Accounts PASSEDBLOWN (>30d):':<35} {number_accounts_passedblown}  "
      f"({number_accounts_passedblown / number_accounts_started * 100:.1f}%)")
print(f"{'Accounts BLOWN (DD breach):':<35} {number_accounts_blown}  "
      f"({number_accounts_blown / number_accounts_started * 100:.1f}%)")
print(f"{'Accounts still ACTIVE:':<35} {number_accounts_alive}  "
      f"({number_accounts_alive / number_accounts_started * 100:.1f}%)")
print("-" * 60)

print("\nPER-ACCOUNT DETAIL:")
print("-" * 60)
for acc in accounts:
    if acc.get('passed', False):
        days = (acc['pass_date'] - acc['start_date']).days
        status = "PASSEDBLOWN ✗" if days > 30 else "PASSED ✓"
        extra = (f"Pass Date: {acc['pass_date'].date()}  "
                 f"Pass P&L: ${acc['pass_pnl']:,.2f}  "
                 f"Duration: {days}d")
    elif not acc['alive']:
        status = "BLOWN  ✗"
        days = (acc['blow_date'] - acc['start_date']).days if acc['blow_date'] else "?"
        days_str = f"{days}d" if isinstance(days, int) else days
        extra = (f"Blow Date: {acc['blow_date'].date() if acc['blow_date'] else 'N/A'}  "
                 f"Final P&L: ${acc['pnl']:,.2f}  "
                 f"Duration: {days_str}")
    else:
        status = "ACTIVE  "
        extra  = f"Current P&L: ${acc['pnl']:,.2f}"
    print(f"  Acc {acc['id']:>3} | {status} | {extra} | Peak Closed P&L: ${acc['peak_closed_pnl']:,.2f}")
print("-" * 60)

# Capital summary
total_capital_deployed = START_CAPITAL * number_accounts_started
total_passed_pnl       = sum(acc['pass_pnl'] for acc in accounts if acc.get('passed', False))
freeze_count           = sum(1 for acc in accounts if acc['freeze_triggered'])

print("\nCAPITAL METRICS")
print("-" * 60)
print(f"{'Total Capital Deployed:':<35} ${total_capital_deployed:,.2f}")
if number_accounts_passed > 0:
    print(f"{'Total P&L from Passed Accounts:':<35} ${total_passed_pnl:,.2f}")
    print(f"{'Avg P&L per Passed Account:':<35} ${total_passed_pnl / number_accounts_passed:,.2f}")
if number_accounts_started > 0:
    print(f"{'Return on Capital (passed only):':<35} {total_passed_pnl / total_capital_deployed * 100:.1f}%")

print(f"\n{'Accounts hit freeze trigger:':<35} {freeze_count}")

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

print("\n" + "=" * 60)
print("\nNOTE: Simulation assumes MAE occurs before MFE in each trade (conservative).")
print(f"Total Capital Deployed across {number_accounts_started} accounts: ${total_capital_deployed:,.2f}")

try:
    plt.show()
except KeyboardInterrupt:
    print("\nScript stopped by user.")