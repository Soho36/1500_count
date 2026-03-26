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

CSV_PATH = "databento_all.csv"

# --- Date range filter ---
START_DATE = "2025-01-01"
END_DATE   = "2025-12-30"

# --- Account start timing (shared by both straight and mirror) ---
START_NEW_ACCOUNT_DAYS  = 30
MIN_DAYS_BETWEEN_STARTS = 1

# ==================================================================
#  STRAIGHT ACCOUNTS CONFIG
# ==================================================================
STRAIGHT_ENABLED        = False
STRAIGHT_MAX_DRAWDOWN   = 2000
STRAIGHT_PROFIT_TARGET  = 3000
STRAIGHT_MAX_ACCOUNTS   = 100

STRAIGHT_START_CAPITAL        = STRAIGHT_MAX_DRAWDOWN
STRAIGHT_DD_FREEZE_TRIGGER    = STRAIGHT_START_CAPITAL + STRAIGHT_MAX_DRAWDOWN + 100
STRAIGHT_FROZEN_DD_FLOOR      = STRAIGHT_START_CAPITAL + 100

# ==================================================================
#  MIRROR ACCOUNTS CONFIG
# ==================================================================
MIRROR_ENABLED          = True
MIRROR_MAX_DRAWDOWN     = 2000
MIRROR_PROFIT_TARGET    = 3000
MIRROR_MAX_ACCOUNTS     = 100

MIRROR_START_CAPITAL      = MIRROR_MAX_DRAWDOWN
MIRROR_DD_FREEZE_TRIGGER  = MIRROR_START_CAPITAL + MIRROR_MAX_DRAWDOWN + 100
MIRROR_FROZEN_DD_FLOOR    = MIRROR_START_CAPITAL + 100

# --- PASSEDBLOWN time threshold (calendar days) ---
PASSEDBLOWN_DAYS = 300

# ==================================================================
#  NEW ACCOUNT TRIGGERS  (same logic for both straight and mirror)
# ==================================================================
# MAX_ACCOUNTS         = 1        # upper bound (overridden per run by STRAIGHT/MIRROR_MAX_ACCOUNTS)
USE_TIME_TRIGGER     = True
USE_PROFIT_TRIGGER   = False
START_IF_PROFIT_THRESHOLD = 1000
USE_DD_TRIGGER       = False
START_IF_DD_THRESHOLD     = 400
RECOVERY_LEVEL       = 0

# ==================================================================
#  PLOTS SWITCHES
# ==================================================================
UNIFIED_EQUITY_AND_DD_PLOTS_3 = True
STARTED_ACCOUNTS_PNL_PLOT     = True
PORTFOLIO_TOTAL_PNL_PLOT      = True
MONTHLY_OUTCOMES_BAR_PLOT     = True


# ======================
#  FUNCTIONS
# ======================

def load_and_preprocess_data(csv_path, start_date=None, end_date=None):
    try:
        df = pd.read_csv(csv_path, sep="\t")
    except Exception as e:
        print("ERROR LOADING CSV FILE:", e)
        exit(1)

    for col in ["PNL", "MAE", "MFE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".").str.strip(),
                errors='coerce'
            ).fillna(0)

    df["Entry_time"] = pd.to_datetime(df["Entry_time"])
    df["Exit_time"]  = pd.to_datetime(df["Exit_time"])

    if start_date:
        df = df[df["Exit_time"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Exit_time"] <= pd.to_datetime(end_date)]

    return df.sort_values("Entry_time").reset_index(drop=True)


def make_mirror_df(df):
    """Flip all trade directions: PNL→-PNL, MAE↔-MFE, MFE↔-MAE."""
    m = df.copy()
    m["PNL"] = -df["PNL"]
    m["MAE"] = -df["MFE"]   # original best case  → mirror worst case
    m["MFE"] = -df["MAE"]   # original worst case → mirror best case
    return m


def compute_prop_style_drawdown(df):
    rows = []
    equity = 0.0
    for _, r in df.sort_values("Entry_time").iterrows():
        rows += [
            {"time": r["Entry_time"], "equity": equity,              "event": "entry"},
            {"time": r["Entry_time"], "equity": equity + r["MAE"],   "event": "mae"},
            {"time": r["Entry_time"], "equity": equity + r["MFE"],   "event": "mfe"},
        ]
        equity += r["PNL"]
        rows.append({"time": r["Exit_time"], "equity": equity, "event": "exit"})

    equity_curve = pd.DataFrame(rows).sort_values("time")
    equity_curve["time"] = pd.to_datetime(equity_curve["time"])

    peak, worst_dd, dd_rows = 0.0, 0.0, []
    for _, r in equity_curve.iterrows():
        peak = max(peak, r["equity"])
        dd   = r["equity"] - peak
        worst_dd = min(worst_dd, dd)
        dd_rows.append({"time": r["time"], "equity": r["equity"],
                        "equity_peak": peak, "trailing_dd": dd,
                        "worst_dd_so_far": worst_dd, "event": r["event"]})

    dd_curve = pd.DataFrame(dd_rows)
    daily = (
        dd_curve
        .assign(Date=pd.to_datetime(dd_curve["time"]).dt.date)
        .groupby("Date")
        .agg(Equity=("equity","last"), Equity_Peak=("equity_peak","max"),
             Equity_Low=("equity","min"), DD_Floating=("trailing_dd","min"))
        .reset_index()
    )
    daily["Date"]        = pd.to_datetime(daily["Date"])
    daily["DD_Floating"] = daily["Equity_Low"] - daily["Equity_Peak"]
    daily["Closed_Peak"] = daily["Equity"].cummax()
    daily["DD_Closed"]   = daily["Equity"] - daily["Closed_Peak"]
    return daily, dd_curve


def create_trade_events_with_priority(df):
    events, cumulative_pnl = [], 0
    for trade_idx, trade in df.iterrows():
        pre = cumulative_pnl
        events.append({'time': trade['Entry_time'], 'trade_idx': trade_idx,
                       'event_type': 'entry', 'priority': 0,
                       'pre_trade_equity': pre, 'mae': trade['MAE'],
                       'mfe': trade['MFE'], 'pnl': trade['PNL'], 'equity_change': 0})
        events.append({'time': trade['Entry_time'] + timedelta(microseconds=1),
                       'trade_idx': trade_idx, 'event_type': 'mae', 'priority': 1,
                       'pre_trade_equity': pre, 'mae': trade['MAE'],
                       'mfe': trade['MFE'], 'pnl': trade['PNL'],
                       'temp_equity': pre + trade['MAE']})
        events.append({'time': trade['Entry_time'] + timedelta(microseconds=2),
                       'trade_idx': trade_idx, 'event_type': 'mfe', 'priority': 2,
                       'pre_trade_equity': pre, 'mae': trade['MAE'],
                       'mfe': trade['MFE'], 'pnl': trade['PNL'],
                       'temp_equity': pre + trade['MFE']})
        cumulative_pnl += trade['PNL']
        events.append({'time': trade['Exit_time'], 'trade_idx': trade_idx,
                       'event_type': 'exit', 'priority': 3,
                       'pre_trade_equity': pre, 'mae': trade['MAE'],
                       'mfe': trade['MFE'], 'pnl': trade['PNL'],
                       'new_equity': cumulative_pnl})

    ev = pd.DataFrame(events).sort_values(['time', 'priority']).reset_index(drop=True)
    return ev


def simulate_accounts(events_df, start_capital, max_accounts,
                      dd_max, dd_freeze_trigger, frozen_dd_floor, profit_target,
                      label_prefix=""):
    """
    Core simulation engine.  label_prefix is '' for straight, 'M' for mirror
    so account IDs print as 1, 2, 3 or 1M, 2M, 3M.
    """
    accounts = []

    event_times = events_df['time'].values
    event_types = events_df['event_type'].values
    event_mae   = events_df['mae'].values
    event_pnl   = events_df['pnl'].values
    total_events = len(events_df)

    last_start_date      = pd.Timestamp(event_times[0])
    waiting_for_recovery = False
    portfolio_pnl_history, num_alive_history, portfolio_times = [], [], []
    account_history_points = []

    def make_account(account_id, start_idx, start_date):
        return {
            'id':           f"{account_id}{label_prefix}",
            'start_idx':    start_idx,
            'start_date':   start_date,
            'equity':       start_capital,
            'pnl':          0,
            'freeze_triggered': False,
            'passed':       False,
            'pass_date':    None,
            'pass_pnl':     None,
            'blow_date':    None,
            'eod_dd_floor':       start_capital - dd_max,
            'eod_peak_closing':   start_capital,
            'eod_closing_equity': start_capital,
            'last_eod_date':      None,
            'peak_closed_pnl':    0.0,
            'alive':              True,
            'current_trade_start_equity': None,
            'last_event_idx':     -1,
        }

    accounts.append(make_account(1, 0, pd.Timestamp(event_times[0])))

    for event_idx in range(total_events):
        current_time = pd.Timestamp(event_times[event_idx])
        current_day  = current_time.date()
        event_type   = event_types[event_idx]

        # ---- EOD floor update ----
        if event_type == 'entry':
            for acc in accounts:
                if not acc['alive']:
                    continue
                if acc['last_eod_date'] is not None and acc['last_eod_date'] < current_day:
                    if not acc['freeze_triggered']:
                        new_peak = max(acc['eod_peak_closing'], acc['eod_closing_equity'])
                        acc['eod_peak_closing'] = new_peak
                        if acc['eod_peak_closing'] >= dd_freeze_trigger:
                            acc['freeze_triggered'] = True
                            acc['eod_dd_floor']     = frozen_dd_floor
                            print(f"Acc {acc['id']} FREEZE on {current_day} | "
                                  f"peak={acc['eod_peak_closing']:.2f}")
                        else:
                            acc['eod_dd_floor'] = max(
                                acc['eod_dd_floor'],
                                acc['eod_peak_closing'] - dd_max
                            )
                    acc['last_eod_date'] = current_day

        active = [a for a in accounts if a['alive'] and a['start_idx'] <= event_idx]

        for acc in active:
            if acc['last_event_idx'] >= event_idx:
                continue
            acc['last_event_idx'] = event_idx

            if event_type == 'entry':
                acc['current_trade_start_equity'] = acc['equity']

            elif event_type == 'mae':
                if acc['current_trade_start_equity'] is None:
                    continue
                temp_eq = acc['current_trade_start_equity'] + event_mae[event_idx]
                if temp_eq <= acc['eod_dd_floor']:
                    acc['alive']     = False
                    acc['equity']    = temp_eq
                    acc['pnl']       = temp_eq - start_capital
                    acc['blow_date'] = current_time
                    print(f"Acc {acc['id']} BLOWN MAE on {current_time} | "
                          f"eq={temp_eq:.2f} floor={acc['eod_dd_floor']:.2f}")
                    account_history_points.append({
                        'time': current_time, 'account_id': acc['id'],
                        'equity': temp_eq, 'pnl': acc['pnl'], 'event': 'blowout_mae'})

            elif event_type == 'mfe':
                pass

            elif event_type == 'exit':
                if acc['current_trade_start_equity'] is None:
                    continue
                new_eq = acc['current_trade_start_equity'] + event_pnl[event_idx]
                acc['equity'] = new_eq
                acc['pnl']    = new_eq - start_capital
                acc['current_trade_start_equity'] = None

                if acc['equity'] <= acc['eod_dd_floor']:
                    acc['alive']     = False
                    acc['blow_date'] = current_time
                    print(f"Acc {acc['id']} BLOWN exit on {current_time} | "
                          f"eq={acc['equity']:.2f} floor={acc['eod_dd_floor']:.2f}")
                    account_history_points.append({
                        'time': current_time, 'account_id': acc['id'],
                        'equity': acc['equity'], 'pnl': acc['pnl'], 'event': 'blowout_exit'})
                    continue

                if acc['pnl'] >= profit_target:
                    acc['alive']     = False
                    acc['passed']    = True
                    acc['pass_date'] = current_time
                    acc['pass_pnl']  = acc['pnl']
                    print(f"Acc {acc['id']} PASSED on {current_time} | P&L=${acc['pnl']:.2f}")
                    account_history_points.append({
                        'time': current_time, 'account_id': acc['id'],
                        'equity': acc['equity'], 'pnl': acc['pnl'], 'event': 'passed'})
                    continue

                acc['eod_closing_equity'] = acc['equity']
                acc['last_eod_date']      = current_day
                acc['peak_closed_pnl']    = max(acc['peak_closed_pnl'], acc['pnl'])
                account_history_points.append({
                    'time': current_time, 'account_id': acc['id'],
                    'equity': acc['equity'], 'pnl': acc['pnl'], 'event': 'exit'})

        if event_type in ['exit', 'mae']:
            portfolio_pnl_history.append(sum(a['pnl'] for a in accounts if a['alive']))
            num_alive_history.append(sum(1 for a in accounts if a['alive']))
            portfolio_times.append(current_time)

        # ---- Start new accounts ----
        if len(accounts) < max_accounts:
            alive_accounts = [a for a in accounts if a['alive']]
            current_dd, can_start, started_due_to_dd = 0, False, False

            if waiting_for_recovery and USE_DD_TRIGGER:
                if current_dd >= RECOVERY_LEVEL:
                    waiting_for_recovery = False
                else:
                    can_start = False

            if not waiting_for_recovery:
                trigger_dd = trigger_profit = trigger_time = False
                if USE_DD_TRIGGER and current_dd <= -START_IF_DD_THRESHOLD:
                    trigger_dd = started_due_to_dd = True
                if USE_PROFIT_TRIGGER and alive_accounts:
                    if alive_accounts[-1]['equity'] - start_capital >= START_IF_PROFIT_THRESHOLD:
                        trigger_profit = True
                if USE_TIME_TRIGGER:
                    if current_time >= last_start_date + pd.Timedelta(days=START_NEW_ACCOUNT_DAYS):
                        trigger_time = True
                if trigger_dd or trigger_profit or trigger_time:
                    can_start = True

            if can_start:
                days_since = (current_time - last_start_date).total_seconds() / 86400
                if days_since >= MIN_DAYS_BETWEEN_STARTS:
                    scheduled = last_start_date + pd.Timedelta(days=START_NEW_ACCOUNT_DAYS)
                    new_acc   = make_account(len(accounts) + 1, event_idx, scheduled)
                    accounts.append(new_acc)
                    print(f"Started Acc {new_acc['id']} scheduled={scheduled} "
                          f"(first event {current_time})")
                    last_start_date      = scheduled
                    waiting_for_recovery = USE_DD_TRIGGER and started_due_to_dd

    print(f"\nSimulation complete [{label_prefix or 'straight'}]: "
          f"{total_events} events, {len(accounts)} accounts.")

    if portfolio_times:
        pnl_s       = pd.Series(portfolio_pnl_history, index=portfolio_times).resample('D').last().ffill()
        alive_s     = pd.Series(num_alive_history,     index=portfolio_times).resample('D').last().ffill()
    else:
        pnl_s = alive_s = pd.Series()

    if account_history_points:
        hist_df     = pd.DataFrame(account_history_points)
        account_pnl = hist_df.pivot_table(index='time', columns='account_id',
                                          values='pnl', aggfunc='last')
        account_pnl.columns = [f'acc_{c}_pnl' for c in account_pnl.columns]
        account_pnl = account_pnl.resample('D').last().ffill()
    else:
        account_pnl = pd.DataFrame()

    return pnl_s, account_pnl, alive_s, accounts


def print_config():
    print("=== Configuration ===")
    print(f"CSV_PATH:                {CSV_PATH}")
    print(f"START_NEW_ACCOUNT_DAYS:  {START_NEW_ACCOUNT_DAYS}")
    print(f"PASSEDBLOWN_DAYS:        {PASSEDBLOWN_DAYS}")
    if STRAIGHT_ENABLED:
        print(f"\n  [STRAIGHT]  MAX_DD={STRAIGHT_MAX_DRAWDOWN}  "
              f"TARGET={STRAIGHT_PROFIT_TARGET}  MAX_ACC={STRAIGHT_MAX_ACCOUNTS}")
    if MIRROR_ENABLED:
        print(f"  [MIRROR]    MAX_DD={MIRROR_MAX_DRAWDOWN}  "
              f"TARGET={MIRROR_PROFIT_TARGET}  MAX_ACC={MIRROR_MAX_ACCOUNTS}")
    print("=====================")


def build_outcome_records(accounts, passedblown_days):
    """Return list of dicts with month / outcome / type for bar chart."""
    records = []
    for acc in accounts:
        if acc.get('passed', False):
            days    = (acc['pass_date'] - acc['start_date']).days
            outcome = 'PASSEDBLOWN' if days > passedblown_days else 'PASSED'
            end_dt  = acc['pass_date']
        elif acc['blow_date'] is not None:
            outcome = 'BLOWN'
            end_dt  = acc['blow_date']
        else:
            continue   # still active
        records.append({'month': end_dt.to_period('M'), 'outcome': outcome})
    return records


def print_summary(accounts, label, start_capital, passedblown_days):
    n_total  = len(accounts)
    n_passed = sum(1 for a in accounts
                   if a.get('passed') and
                   (a['pass_date'] - a['start_date']).days <= passedblown_days)
    n_pb     = sum(1 for a in accounts
                   if a.get('passed') and
                   (a['pass_date'] - a['start_date']).days > passedblown_days)
    n_blown  = sum(1 for a in accounts if not a['alive'] and not a.get('passed'))
    n_alive  = sum(1 for a in accounts if a['alive'])

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY — {label}")
    print(f"{'='*60}")
    print(f"  {'Accounts Started:':<35} {n_total}")
    print(f"  {'Accounts PASSED:':<35} {n_passed}  ({n_passed/n_total*100:.1f}%)")
    print(f"  {'Accounts PASSEDBLOWN (>{passedblown_days}d):':<35} {n_pb}  ({n_pb/n_total*100:.1f}%)")
    print(f"  {'Accounts BLOWN:':<35} {n_blown}  ({n_blown/n_total*100:.1f}%)")
    print(f"  {'Accounts still ACTIVE:':<35} {n_alive}  ({n_alive/n_total*100:.1f}%)")

    print(f"\n  PER-ACCOUNT DETAIL:")
    print(f"  {'-'*56}")
    for acc in accounts:
        if acc.get('passed'):
            days   = (acc['pass_date'] - acc['start_date']).days
            status = "PASSEDBLOWN ✗" if days > passedblown_days else "PASSED ✓      "
            extra  = (f"Pass Date: {acc['pass_date'].date()}  "
                      f"P&L: ${acc['pass_pnl']:,.0f}  Dur: {days}d")
        elif not acc['alive']:
            days   = (acc['blow_date'] - acc['start_date']).days if acc['blow_date'] else "?"
            status = "BLOWN  ✗      "
            extra  = (f"Blow Date: {acc['blow_date'].date() if acc['blow_date'] else 'N/A'}  "
                      f"P&L: ${acc['pnl']:,.0f}  "
                      f"Dur: {days}d" if isinstance(days, int) else f"Dur: {days}")
        else:
            status = "ACTIVE        "
            extra  = f"Current P&L: ${acc['pnl']:,.0f}"
        print(f"  Acc {acc['id']:>4} | {status} | {extra} | "
              f"Peak: ${acc['peak_closed_pnl']:,.0f}")

    total_capital = start_capital * n_total
    passed_pnl    = sum(a['pass_pnl'] for a in accounts if a.get('passed'))
    print(f"\n  CAPITAL METRICS")
    print(f"  {'-'*56}")
    print(f"  {'Total Capital Deployed:':<35} ${total_capital:,.0f}")
    if n_passed > 0:
        print(f"  {'Total P&L — Passed Accs:':<35} ${passed_pnl:,.0f}")
        print(f"  {'Avg P&L per Passed Acc:':<35} ${passed_pnl/n_passed:,.0f}")
    print(f"  {'Return on Capital (passed):':<35} {passed_pnl/total_capital*100:.1f}%")
    freeze_n = sum(1 for a in accounts if a['freeze_triggered'])
    print(f"  {'Accounts hit freeze trigger:':<35} {freeze_n}")


# ======================
#  MAIN EXECUTION
# ======================
print_config()

df = load_and_preprocess_data(CSV_PATH, START_DATE, END_DATE)
print(f"\nLoaded {len(df)} trades  "
      f"{df['Entry_time'].min()} → {df['Exit_time'].max()}")

df_mirror = make_mirror_df(df)
mode      = "Prop Firm — EOD Threshold"

# ---- Drawdown curve (straight only, for the single-strategy plots) ----
print("\nComputing prop-style drawdown...")
daily_data, full_dd_curve = compute_prop_style_drawdown(df)
plot_df = daily_data.copy()

daily_pnl_for_plots = daily_data[["Date","Equity"]].copy().rename(
    columns={"Equity":"PNL_Daily"})
daily_pnl_for_plots["PNL_Daily"] = daily_pnl_for_plots["PNL_Daily"].diff()
daily_pnl_for_plots.iloc[0, daily_pnl_for_plots.columns.get_loc('PNL_Daily')] = \
    daily_data["Equity"].iloc[0]
daily_pnl_for_plots.set_index('Date', inplace=True)

# ---- Create events ----
events_straight = create_trade_events_with_priority(df)
events_mirror   = create_trade_events_with_priority(df_mirror)

# ---- Run simulations ----
straight_accounts = mirror_accounts = []
port_pnl_s = acc_pnl_s = alive_s = pd.Series()
port_pnl_m = acc_pnl_m = alive_m = pd.Series()

if STRAIGHT_ENABLED:
    print(f"\n{'='*60}\nRunning STRAIGHT simulation...\n{'='*60}")
    port_pnl_s, acc_pnl_s, alive_s, straight_accounts = simulate_accounts(
        events_straight,
        start_capital     = STRAIGHT_START_CAPITAL,
        max_accounts      = STRAIGHT_MAX_ACCOUNTS,
        dd_max            = STRAIGHT_MAX_DRAWDOWN,
        dd_freeze_trigger = STRAIGHT_DD_FREEZE_TRIGGER,
        frozen_dd_floor   = STRAIGHT_FROZEN_DD_FLOOR,
        profit_target     = STRAIGHT_PROFIT_TARGET,
        label_prefix      = ""
    )

if MIRROR_ENABLED:
    print(f"\n{'='*60}\nRunning MIRROR simulation...\n{'='*60}")
    port_pnl_m, acc_pnl_m, alive_m, mirror_accounts = simulate_accounts(
        events_mirror,
        start_capital     = MIRROR_START_CAPITAL,
        max_accounts      = MIRROR_MAX_ACCOUNTS,
        dd_max            = MIRROR_MAX_DRAWDOWN,
        dd_freeze_trigger = MIRROR_DD_FREEZE_TRIGGER,
        frozen_dd_floor   = MIRROR_FROZEN_DD_FLOOR,
        profit_target     = MIRROR_PROFIT_TARGET,
        label_prefix      = "M"
    )

# ============================================================
# UNIFIED EQUITY + DD PLOTS  (straight strategy only)
# ============================================================

if UNIFIED_EQUITY_AND_DD_PLOTS_3 and STRAIGHT_ENABLED:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(plot_df["Date"], plot_df["Equity"],      lw=2, label="Equity")
    axes[0].plot(plot_df["Date"], plot_df["Equity_Peak"], lw=1, label="Equity Peak")
    axes[0].plot(plot_df["Date"], plot_df["Equity_Low"],  lw=1, label="Equity Low")
    axes[0].set_title("Equity Curve (Straight Strategy — no DD rules)")
    axes[0].set_ylabel("P&L ($)"); axes[0].grid(True); axes[0].legend()
    axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

    axes[1].plot(plot_df["Date"], plot_df["DD_Closed"], lw=2, label="Closed DD")
    axes[1].axhline(0, lw=0.8)
    axes[1].set_title("Closed Equity Drawdown"); axes[1].set_ylabel("DD ($)")
    axes[1].grid(True); axes[1].legend()
    axes[1].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

    axes[2].plot(plot_df["Date"], plot_df["DD_Floating"], lw=2, label="Floating DD")
    axes[2].axhline(0, lw=0.8)
    axes[2].set_title("Floating Drawdown (MAE/MFE)")
    axes[2].set_xlabel("Date"); axes[2].set_ylabel("DD ($)")
    axes[2].grid(True); axes[2].legend()
    axes[2].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()

# ============================================================
# PORTFOLIO P&L PLOTS  (one per enabled sim)
# ============================================================

def plot_portfolio(port_pnl, acc_pnl_df, accounts, title_suffix):
    if port_pnl.empty or acc_pnl_df.empty:
        return
    all_acc   = acc_pnl_df.sum(axis=1)
    alive_df  = acc_pnl_df.copy()
    for acc in accounts:
        if not acc['alive']:
            col = f'acc_{acc["id"]}_pnl'
            if col in alive_df.columns:
                alive_df[col] = np.nan
    alive_sum = alive_df.sum(axis=1)
    profit_df = acc_pnl_df.copy()
    for col in profit_df.columns:
        profit_df[col] = profit_df[col].clip(lower=0)
    profit_sum = profit_df.sum(axis=1)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(all_acc.index,    all_acc.values,    lw=2, color='orange', label="All accounts (incl. blown)")
    ax.plot(alive_sum.index,  alive_sum.values,  lw=3, color='blue',   label="Alive accounts P&L")
    ax.plot(profit_sum.index, profit_sum.values, lw=2, color='green',  label="Profitable accounts (withdrawable)")
    ax.axhline(0, color='black', lw=0.8, alpha=0.5, linestyle='--')
    ax.set_title(f"Portfolio P&L — {title_suffix}")
    ax.set_ylabel("P&L ($)"); ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3); ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()

if PORTFOLIO_TOTAL_PNL_PLOT:
    if STRAIGHT_ENABLED:
        plot_portfolio(port_pnl_s, acc_pnl_s, straight_accounts,
                       f"Straight Accounts — {mode}")
    if MIRROR_ENABLED:
        plot_portfolio(port_pnl_m, acc_pnl_m, mirror_accounts,
                       f"Mirror Accounts — {mode}")

# ============================================================
# INDIVIDUAL ACCOUNTS P&L PLOTS
# ============================================================

def plot_individual_accounts(acc_pnl_df, accounts, profit_target,
                              frozen_floor, title_suffix):
    if acc_pnl_df.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, col in enumerate(acc_pnl_df.columns[:len(accounts)]):
        ax.plot(acc_pnl_df.index, acc_pnl_df[col],
                alpha=0.7, lw=1.5,
                label=f"Account {accounts[i]['id']}" if i < 10 else None)
    ax.axhline(0, color='black', lw=0.8, alpha=0.5, linestyle='--')
    ax.axhline(frozen_floor,  color='red',   lw=2, linestyle='--', alpha=0.8,
               label=f'Frozen DD floor ({frozen_floor:,.0f})')
    ax.axhline(profit_target, color='green', lw=2, linestyle='--', alpha=0.8,
               label=f'Profit Target ({profit_target:,.0f})')
    ax.set_title(f"Individual Accounts P&L — {title_suffix}")
    ax.set_ylabel("P&L ($)"); ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.grid(True, alpha=0.3); ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()

if STARTED_ACCOUNTS_PNL_PLOT:
    if STRAIGHT_ENABLED:
        plot_individual_accounts(acc_pnl_s, straight_accounts,
                                 STRAIGHT_PROFIT_TARGET, STRAIGHT_FROZEN_DD_FLOOR,
                                 f"Straight — {mode}")
    if MIRROR_ENABLED:
        plot_individual_accounts(acc_pnl_m, mirror_accounts,
                                 MIRROR_PROFIT_TARGET, MIRROR_FROZEN_DD_FLOOR,
                                 f"Mirror — {mode}")

# ============================================================
# MONTHLY OUTCOMES BAR PLOT  (two subplots, shared x-axis)
# ============================================================

if MONTHLY_OUTCOMES_BAR_PLOT:
    rec_s = build_outcome_records(straight_accounts, PASSEDBLOWN_DAYS) if STRAIGHT_ENABLED else []
    rec_m = build_outcome_records(mirror_accounts,   PASSEDBLOWN_DAYS) if MIRROR_ENABLED  else []

    all_months = sorted(set(
        [r['month'] for r in rec_s] + [r['month'] for r in rec_m]
    ))

    def to_monthly_df(records, all_months):
        if not records:
            return pd.DataFrame(0, index=all_months,
                                columns=['PASSED','PASSEDBLOWN','BLOWN'])
        df_o = pd.DataFrame(records)
        mo   = (df_o.groupby(['month','outcome']).size()
                    .unstack(fill_value=0).reindex(all_months, fill_value=0))
        for col in ['PASSED','PASSEDBLOWN','BLOWN']:
            if col not in mo.columns:
                mo[col] = 0
        return mo[['PASSED','PASSEDBLOWN','BLOWN']]

    mo_s = to_monthly_df(rec_s, all_months)
    mo_m = to_monthly_df(rec_m, all_months)

    month_labels = [str(m) for m in all_months]
    x         = np.arange(len(month_labels))
    bw        = 0.25
    colors    = {'PASSED': '#2ecc71', 'PASSEDBLOWN': '#f39c12', 'BLOWN': '#e74c3c'}
    labels    = {'PASSED': 'PASSED ✓',
                 'PASSEDBLOWN': f'PASSEDBLOWN ✗ (>{PASSEDBLOWN_DAYS}d)',
                 'BLOWN': 'BLOWN ✗'}

    n_subplots = sum([STRAIGHT_ENABLED, MIRROR_ENABLED])
    fig_mo, axes_mo = plt.subplots(n_subplots, 1,
                                   figsize=(max(10, len(month_labels) * 1.1),
                                            4.5 * n_subplots),
                                   sharex=True)
    if n_subplots == 1:
        axes_mo = [axes_mo]

    def draw_outcome_bars(ax, mo_df, title):
        offsets = {'PASSED': -bw, 'PASSEDBLOWN': 0, 'BLOWN': bw}
        for outcome, offset in offsets.items():
            bars = ax.bar(x + offset, mo_df[outcome],
                          width=bw, color=colors[outcome],
                          edgecolor='white', lw=0.6, label=labels[outcome])
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                            str(int(h)), ha='center', va='bottom',
                            fontsize=9, fontweight='bold')
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Number of Accounts")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)

    subplot_idx = 0
    if STRAIGHT_ENABLED:
        draw_outcome_bars(
            axes_mo[subplot_idx], mo_s,
            f"Straight Accounts — Monthly Outcomes  "
            f"(Target: ${STRAIGHT_PROFIT_TARGET:,.0f}  MaxDD: ${STRAIGHT_MAX_DRAWDOWN:,.0f})"
        )
        subplot_idx += 1
    if MIRROR_ENABLED:
        draw_outcome_bars(
            axes_mo[subplot_idx], mo_m,
            f"Mirror Accounts — Monthly Outcomes  "
            f"(Target: ${MIRROR_PROFIT_TARGET:,.0f}  MaxDD: ${MIRROR_MAX_DRAWDOWN:,.0f})"
        )

    axes_mo[-1].set_xticks(x)
    axes_mo[-1].set_xticklabels(month_labels, rotation=45, ha='right')
    axes_mo[-1].set_xlabel("Month (outcome date)")
    fig_mo.suptitle(
        f"Monthly Account Outcomes — {mode}  |  "
        f"New account every {START_NEW_ACCOUNT_DAYS}d",
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()

# ======================
#  STATISTICS
# ======================
monthly_pnl = daily_pnl_for_plots.resample('M')['PNL_Daily'].sum()
yearly_pnl  = daily_pnl_for_plots.resample('Y')['PNL_Daily'].sum()

if STRAIGHT_ENABLED:
    print_summary(straight_accounts, "STRAIGHT ACCOUNTS",
                  STRAIGHT_START_CAPITAL, PASSEDBLOWN_DAYS)

if MIRROR_ENABLED:
    print_summary(mirror_accounts, "MIRROR ACCOUNTS",
                  MIRROR_START_CAPITAL, PASSEDBLOWN_DAYS)

print("\n" + "="*60)
print("SINGLE ACCOUNT STRATEGY PERFORMANCE (straight, no DD rules)")
print("="*60)
if not monthly_pnl.empty:
    print(f"  {'Monthly Total:':<30} ${monthly_pnl.sum():,.0f}")
    print(f"  {'Monthly Win Rate:':<30} "
          f"{(monthly_pnl>0).sum()/len(monthly_pnl)*100:.1f}%  "
          f"({(monthly_pnl>0).sum()}/{len(monthly_pnl)})")
    print(f"  {'Best Month:':<30} ${monthly_pnl.max():,.0f}")
    print(f"  {'Avg Month:':<30} ${monthly_pnl.mean():,.0f}")
    print(f"  {'Worst Month:':<30} ${monthly_pnl.min():,.0f}")
if not yearly_pnl.empty:
    print(f"  {'Yearly Total:':<30} ${yearly_pnl.sum():,.0f}")
    print(f"  {'Yearly Win Rate:':<30} "
          f"{(yearly_pnl>0).sum()/len(yearly_pnl)*100:.1f}%  "
          f"({(yearly_pnl>0).sum()}/{len(yearly_pnl)})")

print("\nNOTE: MAE assumed before MFE (conservative). "
      "Mirror trades are fully inverted (PNL/MAE/MFE all flipped).")

try:
    plt.show()
except KeyboardInterrupt:
    print("\nStopped by user.")