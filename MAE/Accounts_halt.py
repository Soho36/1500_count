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

CSV_PATH = "databento_all.csv"

# --- Drawdown settings ---
MAX_DRAWDOWN = 2000
START_CAPITAL = MAX_DRAWDOWN
EQUITY_DD_FREEZE_TRIGGER = START_CAPITAL + MAX_DRAWDOWN + 100
FROZEN_DD_FLOOR = START_CAPITAL + 100

# --- Date range filter ---
START_DATE = "2020-02-01"
END_DATE = None

# ==================================================================
# --- Simulation Mode ---
# Exactly ONE of these should be True
# ==================================================================
USE_TRAILING_DD  = False   # Live trailing: floor trails highest intraday balance
USE_EOD_DRAWDOWN = True    # EOD threshold: floor is set once at market close each day

# ==================================================================
# --- Halt settings ---
# ==================================================================
# HALT_ENABLED : set to True to activate periodic halt cycles
# HALT_TRADE_DAYS : number of calendar E-R days the account trades before halting
# HALT_PAUSE_DAYS : number of calendar E-R days the account is paused (no trades)
#
# Cycle repeats forever: TRADE → HALT → TRADE → HALT → ...
# Clock starts from each account's own start_date.
# During halt the EOD floor stays frozen at the last closing floor value.
# ==================================================================
HALT_ENABLED    = False
HALT_TRADE_DAYS = 10   # trade for this many calendar E-R days
HALT_PAUSE_DAYS = 10   # then pause for this many calendar E-R days

# ==================================================================
# --- New account start triggers ---
# ==================================================================
MAX_ACCOUNTS = 1
USE_TIME_TRIGGER = True
TIME_TRIGGER_DAYS = 15
USE_PROFIT_TRIGGER = False
START_IF_PROFIT_THRESHOLD = 1000
USE_DD_TRIGGER = False
START_IF_DD_THRESHOLD = 400
RECOVERY_LEVEL = 0
MIN_DAYS_BETWEEN_STARTS = 1

SHOW_PORTFOLIO_TOTAL_PNL = True
SHOW_DD_PLOT = True

# ==================================================================
#  SIMULATION ASSUMPTIONS
# ==================================================================
"""
HALT MECHANIC:
  Each account has an independent halt cycle keyed to its own start_date.
  "Calendar E-R days" means we count every calendar day (Mon-Fri market
  days that exist in the data range), regardless of whether trades occurred.

  cycle_day = (current_date - account.start_date).days  (0-based)
  phase      = cycle_day % (HALT_TRADE_DAYS + HALT_PAUSE_DAYS)
  if phase < HALT_TRADE_DAYS  → account is ACTIVE  (trades processed normally)
  if phase >= HALT_TRADE_DAYS → account is HALTED   (all events skipped)

  EOD floor during halt: frozen at the floor value from the last active session.
  The floor does NOT update while halted because no new closing equity exists.
"""


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
            df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df["Entry_time"] = pd.to_datetime(df["Entry_time"])
    df["Exit_time"]  = pd.to_datetime(df["Exit_time"])

    if start_date:
        df = df[df["Exit_time"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Exit_time"] <= pd.to_datetime(end_date)]

    return df.sort_values("Entry_time").reset_index(drop=True)


def compute_prop_style_drawdown(df):
    rows   = []
    equity = 0.0

    for _, r in df.sort_values("Entry_time").iterrows():
        rows.append({"time": r["Entry_time"], "equity": equity,            "event": "entry"})
        rows.append({"time": r["Entry_time"], "equity": equity + r["MAE"], "event": "mae"})
        rows.append({"time": r["Entry_time"], "equity": equity + r["MFE"], "event": "mfe"})
        equity += r["PNL"]
        rows.append({"time": r["Exit_time"],  "equity": equity,            "event": "exit"})

    equity_curve             = pd.DataFrame(rows).sort_values("time")
    equity_curve["time"]     = pd.to_datetime(equity_curve["time"])

    equity_peak = 0.0
    dd_rows     = []
    for _, r in equity_curve.iterrows():
        equity_peak = max(equity_peak, r["equity"])
        dd_rows.append({
            "time": r["time"], "equity": r["equity"],
            "equity_peak": equity_peak,
            "trailing_dd": r["equity"] - equity_peak,
            "event": r["event"]
        })

    dd_curve = pd.DataFrame(dd_rows)

    daily_data = (
        dd_curve
        .assign(Date=pd.to_datetime(dd_curve["time"]).dt.date)
        .groupby("Date")
        .agg(Equity=("equity","last"), Equity_Peak=("equity_peak","max"),
             Equity_Low=("equity","min"), DD_Floating=("trailing_dd","min"))
        .reset_index()
    )
    daily_data["Date"]        = pd.to_datetime(daily_data["Date"])
    daily_data["DD_Floating"] = daily_data["Equity_Low"]   - daily_data["Equity_Peak"]
    daily_data["Closed_Peak"] = daily_data["Equity"].cummax()
    daily_data["DD_Closed"]   = daily_data["Equity"]       - daily_data["Closed_Peak"]

    return daily_data, dd_curve


def create_trade_events_with_priority(df):
    events      = []
    cum_pnl     = 0

    for trade_idx, trade in df.iterrows():
        pre = cum_pnl
        events.append({"time": trade["Entry_time"],
                        "event_type": "entry", "priority": 0,
                        "pre_trade_equity": pre,
                        "mae": trade["MAE"], "mfe": trade["MFE"], "pnl": trade["PNL"]})
        events.append({"time": trade["Entry_time"] + timedelta(microseconds=1),
                        "event_type": "mae", "priority": 1,
                        "pre_trade_equity": pre,
                        "mae": trade["MAE"], "mfe": trade["MFE"], "pnl": trade["PNL"]})
        events.append({"time": trade["Entry_time"] + timedelta(microseconds=2),
                        "event_type": "mfe", "priority": 2,
                        "pre_trade_equity": pre,
                        "mae": trade["MAE"], "mfe": trade["MFE"], "pnl": trade["PNL"]})
        cum_pnl += trade["PNL"]
        events.append({"time": trade["Exit_time"],
                        "event_type": "exit", "priority": 3,
                        "pre_trade_equity": pre,
                        "mae": trade["MAE"], "mfe": trade["MFE"], "pnl": trade["PNL"]})

    return (pd.DataFrame(events)
              .sort_values(["time", "priority"])
              .reset_index(drop=True))


def is_account_halted(acc, current_date):
    """
    Returns True if the account is in its HALT phase on current_date.

    Logic:
      cycle_length = HALT_TRADE_DAYS + HALT_PAUSE_DAYS
      days_since_start = (current_date - acc['start_date'].date()).days
      phase = days_since_start % cycle_length
      halted when phase >= HALT_TRADE_DAYS
    """
    if not HALT_ENABLED:
        return False
    cycle_length     = HALT_TRADE_DAYS + HALT_PAUSE_DAYS
    days_since_start = (current_date - acc["start_date"].date()).days
    phase            = days_since_start % cycle_length
    return phase >= HALT_TRADE_DAYS


def simulate_accounts_with_prop_dd_optimized(events_df, start_capital, max_accounts):
    if USE_TRAILING_DD and USE_EOD_DRAWDOWN:
        raise ValueError("Set exactly one of USE_TRAILING_DD / USE_EOD_DRAWDOWN to True.")

    event_times = events_df["time"].values
    event_types = events_df["event_type"].values
    event_mae   = events_df["mae"].values
    event_mfe   = events_df["mfe"].values
    event_pnl   = events_df["pnl"].values
    total_events = len(events_df)

    def make_account(account_id, start_idx, start_date):
        return {
            "id":                        account_id,
            "start_idx":                 start_idx,
            "start_date":                start_date,
            "equity":                    start_capital,
            "pnl":                       0,
            # Trailing DD state
            "peak":                      start_capital,
            "freeze_triggered":          False,
            # EOD DD state
            "eod_dd_floor":              start_capital - MAX_DRAWDOWN,
            "eod_closing_equity":        start_capital,
            "eod_peak_closing":          start_capital,
            "last_eod_date":             None,
            # Shared
            "alive":                     True,
            "current_trade_start_equity": None,
            "last_event_idx":            -1,
        }

    accounts             = [make_account(1, 0, pd.Timestamp(event_times[0]))]
    last_start_date      = pd.Timestamp(event_times[0])
    waiting_for_recovery = False

    portfolio_pnl_history = []
    num_alive_history     = []
    portfolio_times       = []
    account_history_points = []

    for event_idx in range(total_events):
        current_time = pd.Timestamp(event_times[event_idx])
        current_day  = current_time.date()
        event_type   = event_types[event_idx]

        active_accounts = [
            acc for acc in accounts
            if acc["alive"] and acc["start_idx"] <= event_idx
        ]

        for acc in active_accounts:
            if acc["last_event_idx"] >= event_idx:
                continue
            acc["last_event_idx"] = event_idx

            # ----------------------------------------------------------
            # HALT CHECK — skip all trade processing if halted.
            # EOD floor does not update during halt (frozen at last value).
            # ----------------------------------------------------------
            if is_account_halted(acc, current_day) and event_type == "entry":
                continue

            # ── ENTRY ──────────────────────────────────────────────────
            if event_type == "entry":
                acc["current_trade_start_equity"] = acc["equity"]

            # ── MAE ────────────────────────────────────────────────────
            elif event_type == "mae":
                if acc["current_trade_start_equity"] is None:
                    continue
                temp_equity = acc["current_trade_start_equity"] + event_mae[event_idx]

                if USE_TRAILING_DD:
                    floor = FROZEN_DD_FLOOR if acc["freeze_triggered"] else acc["peak"] - MAX_DRAWDOWN
                    if temp_equity <= floor:
                        acc["alive"] = False
                        print(f"Acc {acc['id']} BLOWN intraday (Trailing DD) {current_time} | "
                              f"equity={temp_equity:.2f} floor={floor:.2f}")
                        account_history_points.append({
                            "time": current_time, "account_id": acc["id"],
                            "equity": temp_equity, "pnl": acc["pnl"], "event": "blowout_mae"})

                elif USE_EOD_DRAWDOWN:
                    floor = acc["eod_dd_floor"]
                    if temp_equity <= floor:
                        acc["alive"] = False
                        print(f"Acc {acc['id']} BLOWN intraday (EOD floor) {current_time} | "
                              f"equity={temp_equity:.2f} floor={floor:.2f}")
                        account_history_points.append({
                            "time": current_time, "account_id": acc["id"],
                            "equity": temp_equity, "pnl": acc["pnl"], "event": "blowout_mae"})

            # ── MFE ────────────────────────────────────────────────────
            elif event_type == "mfe":
                if acc["current_trade_start_equity"] is None or not acc["alive"]:
                    continue
                if USE_TRAILING_DD:
                    temp_equity = acc["current_trade_start_equity"] + event_mfe[event_idx]
                    if temp_equity > acc["peak"]:
                        acc["peak"] = temp_equity
                        if acc["peak"] >= EQUITY_DD_FREEZE_TRIGGER:
                            acc["freeze_triggered"] = True

            # ── EXIT ───────────────────────────────────────────────────
            elif event_type == "exit":
                if acc["current_trade_start_equity"] is None:
                    continue

                new_equity = acc["current_trade_start_equity"] + event_pnl[event_idx]
                acc["equity"] = new_equity
                acc["pnl"]    = new_equity - start_capital
                acc["current_trade_start_equity"] = None

                if USE_TRAILING_DD:
                    if acc["equity"] > acc["peak"]:
                        acc["peak"] = acc["equity"]
                        if acc["peak"] >= EQUITY_DD_FREEZE_TRIGGER:
                            acc["freeze_triggered"] = True
                    floor = FROZEN_DD_FLOOR if acc["freeze_triggered"] else acc["peak"] - MAX_DRAWDOWN
                    if acc["equity"] <= floor:
                        acc["alive"] = False
                        print(f"Acc {acc['id']} BLOWN exit (Trailing DD) {current_time} | "
                              f"equity={acc['equity']:.2f} floor={floor:.2f}")
                        account_history_points.append({
                            "time": current_time, "account_id": acc["id"],
                            "equity": acc["equity"], "pnl": acc["pnl"], "event": "blowout_exit"})
                        continue

                elif USE_EOD_DRAWDOWN:
                    floor = acc["eod_dd_floor"]
                    if acc["equity"] <= floor:
                        acc["alive"] = False
                        print(f"Acc {acc['id']} BLOWN exit (EOD floor) {current_time} | "
                              f"equity={acc['equity']:.2f} floor={floor:.2f}")
                        account_history_points.append({
                            "time": current_time, "account_id": acc["id"],
                            "equity": acc["equity"], "pnl": acc["pnl"], "event": "blowout_exit"})
                        continue

                # Store last-close for EOD floor recalc (only when active/not halted)
                if USE_EOD_DRAWDOWN and acc["alive"]:
                    acc["eod_closing_equity"] = acc["equity"]
                    acc["last_eod_date"]      = current_day

                account_history_points.append({
                    "time": current_time, "account_id": acc["id"],
                    "equity": acc["equity"], "pnl": acc["pnl"], "event": "exit"})

        # ── EOD FLOOR UPDATE (fires on first entry of a new day) ───────
        # Only updates accounts that are NOT halted on this new day.
        if USE_EOD_DRAWDOWN and event_type == "entry":
            for acc in accounts:
                if not acc["alive"]:
                    continue
                if acc["last_eod_date"] is not None and acc["last_eod_date"] < current_day:
                    # Skip floor update if the account is halted today —
                    # floor stays frozen at the value from the last active session.
                    if is_account_halted(acc, current_day):
                        # Still bump last_eod_date so we re-check tomorrow
                        acc["last_eod_date"] = current_day
                        continue

                    if not acc["freeze_triggered"]:
                        closing_eq              = acc["eod_closing_equity"]
                        acc["eod_peak_closing"] = max(acc["eod_peak_closing"], closing_eq)
                        if acc["eod_peak_closing"] >= EQUITY_DD_FREEZE_TRIGGER:
                            acc["freeze_triggered"] = True
                            acc["eod_dd_floor"]     = FROZEN_DD_FLOOR
                            print(f"Acc {acc['id']} FREEZE {current_day} | "
                                  f"peak_closing={acc['eod_peak_closing']:.2f}")
                        else:
                            acc["eod_dd_floor"] = closing_eq - MAX_DRAWDOWN
                    acc["last_eod_date"] = current_day

        # ── PORTFOLIO SNAPSHOT ─────────────────────────────────────────
        if event_type in ["exit", "mae"]:
            portfolio_pnl_history.append(sum(a["pnl"] for a in accounts if a["alive"]))
            num_alive_history.append(sum(1 for a in accounts if a["alive"]))
            portfolio_times.append(current_time)

        # ── START NEW ACCOUNTS ─────────────────────────────────────────
        if len(accounts) < max_accounts:
            alive_accounts = [a for a in accounts if a["alive"]]
            current_dd     = (min(a["equity"] - a["peak"] for a in alive_accounts)
                              if alive_accounts and USE_TRAILING_DD else 0)

            can_start         = False
            started_due_to_dd = False

            if waiting_for_recovery and USE_DD_TRIGGER:
                if current_dd >= RECOVERY_LEVEL:
                    waiting_for_recovery = False
                else:
                    can_start = False

            if not waiting_for_recovery:
                trigger_dd     = USE_DD_TRIGGER and current_dd <= -START_IF_DD_THRESHOLD
                trigger_profit = (USE_PROFIT_TRIGGER and alive_accounts and
                                  alive_accounts[-1]["equity"] - start_capital >= START_IF_PROFIT_THRESHOLD)
                trigger_time   = (USE_TIME_TRIGGER and
                                  current_time >= last_start_date + pd.Timedelta(days=TIME_TRIGGER_DAYS))
                if trigger_dd:
                    started_due_to_dd = True
                can_start = trigger_dd or trigger_profit or trigger_time

            if can_start:
                days_since = (current_time - last_start_date).total_seconds() / 86400
                if days_since >= MIN_DAYS_BETWEEN_STARTS:
                    scheduled = last_start_date + pd.Timedelta(days=TIME_TRIGGER_DAYS)
                    accounts.append(make_account(len(accounts) + 1, event_idx, scheduled))
                    print(f"Started Acc {len(accounts)} scheduled {scheduled} (event @ {current_time})")
                    last_start_date      = scheduled
                    waiting_for_recovery = USE_DD_TRIGGER and started_due_to_dd

    print(f"\nDone: {total_events} events, {len(accounts)} accounts.")

    if portfolio_times:
        portfolio_pnl_daily = (pd.Series(portfolio_pnl_history, index=portfolio_times)
                               .resample("D").last().ffill())
        num_alive_daily     = (pd.Series(num_alive_history, index=portfolio_times)
                               .resample("D").last().ffill())
    else:
        portfolio_pnl_daily = pd.Series()
        num_alive_daily     = pd.Series()

    if account_history_points:
        hdf         = pd.DataFrame(account_history_points)
        account_pnl = (hdf.pivot_table(index="time", columns="account_id",
                                       values="pnl", aggfunc="last")
                          .rename(columns=lambda c: f"acc_{c}_pnl")
                          .resample("D").last().ffill())
    else:
        account_pnl = pd.DataFrame()

    return portfolio_pnl_daily, account_pnl, num_alive_daily, accounts


def print_config():
    mode = "Trailing DD" if USE_TRAILING_DD else ("EOD Threshold" if USE_EOD_DRAWDOWN else "Closed Equity")
    print("=== Configuration ===")
    print(f"  CSV_PATH:          {CSV_PATH}")
    print(f"  Mode:              {mode}")
    print(f"  MAX_DRAWDOWN:      {MAX_DRAWDOWN}")
    print(f"  DD_FREEZE_TRIGGER: {EQUITY_DD_FREEZE_TRIGGER}")
    print(f"  FROZEN_DD_FLOOR:   {FROZEN_DD_FLOOR}")
    if START_DATE: print(f"  START_DATE:        {START_DATE}")
    if END_DATE:   print(f"  END_DATE:          {END_DATE}")
    print(f"  MAX_ACCOUNTS:      {MAX_ACCOUNTS}")
    print(f"  TIME_TRIGGER_DAYS: {TIME_TRIGGER_DAYS}")
    if HALT_ENABLED:
        print(f"  HALT:              ENABLED — trade {HALT_TRADE_DAYS}d / pause {HALT_PAUSE_DAYS}d (repeating)")
    else:
        print(f"  HALT:              disabled")
    print("=====================")


# ======================
#  MAIN EXECUTION
# ======================
print_config()

df = load_and_preprocess_data(CSV_PATH, START_DATE, END_DATE)
print(f"\nLoaded {len(df)} trades  {df['Entry_time'].min()} → {df['Exit_time'].max()}")

if USE_TRAILING_DD:
    mode = "Prop Firm — Live Trailing DD"
elif USE_EOD_DRAWDOWN:
    mode = "Prop Firm — EOD Threshold"
else:
    mode = "Closed Equity"

print("\n" + "=" * 60)
print("SIMULATION MODE:", mode)
if HALT_ENABLED:
    print(f"HALT: trade {HALT_TRADE_DAYS} days → pause {HALT_PAUSE_DAYS} days → repeat")
print("=" * 60)

# Single-account curves (no halt applied — shows raw strategy)
daily_data, _ = compute_prop_style_drawdown(df)
plot_df        = daily_data.copy()

daily_pnl_for_plots = daily_data[["Date", "Equity"]].copy().rename(columns={"Equity": "PNL_Daily"})
daily_pnl_for_plots["PNL_Daily"] = daily_pnl_for_plots["PNL_Daily"].diff()
daily_pnl_for_plots.iloc[0, daily_pnl_for_plots.columns.get_loc("PNL_Daily")] = daily_data["Equity"].iloc[0]
daily_pnl_for_plots.set_index("Date", inplace=True)

events_df = create_trade_events_with_priority(df)
print(f"Events: {len(events_df)} | {events_df['event_type'].value_counts().to_dict()}")

portfolio_pnl, acc_pnl_df, num_alive_df, accounts = simulate_accounts_with_prop_dd_optimized(
    events_df, START_CAPITAL, MAX_ACCOUNTS
)
number_accounts_started = len(accounts)

# ============================================================
# PLOT 1: Single-account equity + DD panels
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(plot_df["Date"], plot_df["Equity"],      lw=2, label="Equity")
axes[0].plot(plot_df["Date"], plot_df["Equity_Peak"], lw=1, label="Equity_Peak")
axes[0].plot(plot_df["Date"], plot_df["Equity_Low"],  lw=1, label="Equity_Low")
axes[0].set_title("Equity Curve (Single Account Strategy — no halt)")
axes[0].set_ylabel("Equity ($)")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(plot_df["Date"], plot_df["DD_Closed"],   lw=2, label="Closed DD")
axes[1].axhline(0, lw=0.8)
axes[1].set_title("Closed Equity Drawdown")
axes[1].set_ylabel("Drawdown ($)")
axes[1].grid(True)
axes[1].legend()

axes[2].plot(plot_df["Date"], plot_df["DD_Floating"], lw=2, label="Floating DD")
axes[2].axhline(0, lw=0.8)
axes[2].set_title("Floating Drawdown (Intraday MAE/MFE)")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Drawdown ($)")
axes[2].grid(True)
axes[2].legend()

axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()

# ============================================================
# PLOT 2: Portfolio P&L comparison
# ============================================================
portfolio_all_accounts = acc_pnl_df.sum(axis=1)

portfolio_alive_accounts = acc_pnl_df.copy()
for acc in accounts:
    if not acc["alive"]:
        col = f'acc_{acc["id"]}_pnl'
        if col in portfolio_alive_accounts.columns:
            portfolio_alive_accounts[col] = np.nan
portfolio_alive_accounts = portfolio_alive_accounts.sum(axis=1)

portfolio_profitable_accounts = acc_pnl_df.clip(lower=0).sum(axis=1)

if SHOW_PORTFOLIO_TOTAL_PNL and not portfolio_pnl.empty:
    fig_portfolio, ax_portfolio = plt.subplots(figsize=(14, 6))

    ax_portfolio.plot(portfolio_all_accounts.index, portfolio_all_accounts.values,
                      lw=2, color="orange",
                      label="Strategy P&L — all accounts (alive, blown, in loss)")
    ax_portfolio.plot(portfolio_alive_accounts.index, portfolio_alive_accounts.values,
                      lw=3, color="blue",
                      label="Portfolio P&L — alive accounts")
    ax_portfolio.plot(portfolio_profitable_accounts.index, portfolio_profitable_accounts.values,
                      lw=2, color="green",
                      label="Withdrawable P&L — profitable accounts only")

    ax_portfolio.axhline(y=0, color="black", lw=0.8, alpha=0.5, linestyle="--")
    ax_portfolio.set_title(f"Portfolio P&L Comparison — {mode}"
                           + (f" | Halt {HALT_TRADE_DAYS}d/{HALT_PAUSE_DAYS}d" if HALT_ENABLED else ""))
    ax_portfolio.set_ylabel("P&L ($)")
    ax_portfolio.set_xlabel("Date")
    ax_portfolio.grid(True, alpha=0.3)
    ax_portfolio.legend()
    ax_portfolio.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_portfolio.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax_portfolio.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()

# ============================================================
# PLOT 3: Individual accounts P&L
# ============================================================
if not acc_pnl_df.empty:
    fig_acc, ax_acc = plt.subplots(figsize=(14, 6))

    for i, col in enumerate(acc_pnl_df.columns[:number_accounts_started]):
        ax_acc.plot(acc_pnl_df.index, acc_pnl_df[col],
                    alpha=0.7, lw=1.5,
                    label=f"Account {i+1}" if i < 10 else None)

    ax_acc.axhline(y=0, color="black", lw=0.8, alpha=0.5, linestyle="--")
    ax_acc.axhline(
        y=FROZEN_DD_FLOOR,
        color='red', linewidth=2, linestyle='--', alpha=0.8,
        label=f'Frozen DD floor (equity={FROZEN_DD_FLOOR})'
    )

    ax_acc.set_title(f"Individual Accounts P&L — {mode}"
                     + (f" | Halt {HALT_TRADE_DAYS}d/{HALT_PAUSE_DAYS}d" if HALT_ENABLED else ""))
    ax_acc.set_ylabel("P&L ($)")
    ax_acc.set_xlabel("Date")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()
    ax_acc.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_acc.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax_acc.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()

# ============================================================
# PLOT 4: Active accounts over time
# ============================================================
if not num_alive_df.empty:
    fig3, ax5 = plt.subplots(1, 1, figsize=(14, 6))

    ax5.plot(num_alive_df.index, num_alive_df.values,
             color="purple", lw=3, label="Alive Accounts")
    ax5.fill_between(num_alive_df.index, num_alive_df.values, 0,
                     color="purple", alpha=0.2)
    ax5.axhline(y=number_accounts_started, color="gray",  lw=1.2, linestyle="--",
                label=f"Total started: {number_accounts_started}")
    ax5.axhline(y=num_alive_df.iloc[-1],   color="green", lw=1.2, linestyle="--",
                label=f"Final alive: {int(num_alive_df.iloc[-1])}")

    ax5.set_title(f"Active Accounts Over Time — {mode}"
                  + (f" | Halt {HALT_TRADE_DAYS}d/{HALT_PAUSE_DAYS}d" if HALT_ENABLED else ""))
    ax5.set_ylabel("# Accounts")
    ax5.set_xlabel("Date")
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="upper left")
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()

# ============================================================
# BAR CHARTS helper
# ============================================================
def bar_chart(series, title, xlabel, fig_size=(14, 6), yearly=False, color_bg="wheat"):
    fig, ax = plt.subplots(figsize=fig_size)
    clrs    = ["green" if v >= 0 else "red" for v in series.values]
    x_vals  = series.index.year if yearly else series.index
    kw      = {"color": clrs, "alpha": 0.7, "edgecolor": "black", "linewidth": 0.4}
    if yearly:
        kw["width"] = 0.6
    ax.bar(x_vals, series.values, **kw)
    ax.axhline(0, color="black", lw=0.8, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("P&L ($)")
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels for yearly
    if yearly:
        for bar in ax.patches:
            h    = bar.get_height()
            ypos = h + abs(series.values).max() * 0.02 if h >= 0 else h - abs(series.values).max() * 0.02
            ax.text(bar.get_x() + bar.get_width() / 2, ypos, f"${h:,.0f}",
                    ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=9, fontweight="bold")

    total    = series.sum()
    pos      = (series > 0).sum()
    win_rate = pos / len(series) * 100 if len(series) else 0
    avg      = series.mean()
    ax.text(0.02, 0.98,
            f"Total: ${total:,.0f}  |  Avg: ${avg:,.0f}  |  Win Rate: {win_rate:.1f}% ({pos}/{len(series)})",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor=color_bg, alpha=0.5))
    return fig, ax

# ── Single-account bar charts ─────────────────────────────────────────
daily_pnl_plot = daily_pnl_for_plots["PNL_Daily"].copy()
daily_pnl_plot.index = pd.to_datetime(daily_pnl_plot.index)

fig_d, ax_d = bar_chart(daily_pnl_plot, "Single Account — Daily P&L", "Date", fig_size=(16, 6))
ax_d.xaxis.set_major_locator(ticker.MaxNLocator(20))
ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax_d.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
plt.setp(ax_d.xaxis.get_majorticklabels(), rotation=60)
plt.tight_layout()

monthly_pnl = daily_pnl_plot.resample("M").sum()
fig_m, ax_m = bar_chart(monthly_pnl, "Single Account — Monthly P&L", "Month")
ax_m.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_m.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
for bar in ax_m.patches:
    h    = bar.get_height()
    ypos = h + abs(monthly_pnl.values).max() * 0.015 if h >= 0 else h - abs(monthly_pnl.values).max() * 0.015
    ax_m.text(bar.get_x() + bar.get_width() / 2, ypos, f"${h:,.0f}",
              ha="center", va="bottom" if h >= 0 else "top", fontsize=8, rotation=45)
plt.setp(ax_m.xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()

yearly_pnl = daily_pnl_plot.resample("Y").sum()
bar_chart(yearly_pnl, "Single Account — Yearly P&L", "Year", yearly=True)
plt.tight_layout()

# ── Portfolio bar charts ──────────────────────────────────────────────
if not portfolio_pnl.empty:
    port_daily   = portfolio_pnl.diff().fillna(portfolio_pnl.iloc[0])
    port_monthly = port_daily.resample("M").sum()
    port_yearly  = port_daily.resample("Y").sum()

    suffix = f" — {mode}" + (f" | Halt {HALT_TRADE_DAYS}d/{HALT_PAUSE_DAYS}d" if HALT_ENABLED else "")

    fig_pm, ax_pm = bar_chart(port_monthly, f"Portfolio — Monthly P&L{suffix}", "Month",
                              color_bg="lightblue")
    ax_pm.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_pm.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    for bar in ax_pm.patches:
        h    = bar.get_height()
        ypos = h + abs(port_monthly.values).max() * 0.015 if h >= 0 else h - abs(port_monthly.values).max() * 0.015
        ax_pm.text(bar.get_x() + bar.get_width() / 2, ypos, f"${h:,.0f}",
                   ha="center", va="bottom" if h >= 0 else "top", fontsize=8, rotation=45)
    plt.setp(ax_pm.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()

    bar_chart(port_yearly, f"Portfolio — Yearly P&L{suffix}", "Year",
              yearly=True, color_bg="lightblue")
    plt.tight_layout()

# ======================
#  STATISTICS
# ======================
n_alive  = sum(1 for a in accounts if a["alive"])
n_blown  = number_accounts_started - n_alive
n_frozen = sum(1 for a in accounts if a["freeze_triggered"])
strategy_total_pnl       = sum(a["pnl"] for a in accounts)
portfolio_alive_pnl      = sum(a["pnl"] for a in accounts if a["alive"])
portfolio_profitable_pnl = sum(a["pnl"] for a in accounts if a["alive"] and a["pnl"] > 0)
total_capital_deployed   = START_CAPITAL * number_accounts_started
final_portfolio_pnl      = portfolio_pnl.iloc[-1] if not portfolio_pnl.empty else 0

print("\n" + "=" * 60)
print("SIMULATION RESULTS")
print("=" * 60)

print("\nFINAL P&L PER ACCOUNT:")
print("-" * 60)
for acc in accounts:
    status = "ALIVE" if acc["alive"] else "BLOWN ⬤"
    print(f"  Acc {acc['id']:>3} | {status:<8} | P&L: ${acc['pnl']:>8.2f}")

print("-" * 60)
print(f"\n  Mode:              {mode}")
if HALT_ENABLED:
    print(f"  Halt cycle:        trade {HALT_TRADE_DAYS}d / pause {HALT_PAUSE_DAYS}d (repeating)")

print("\nPNL OVERVIEW")
print("-" * 60)
print(f"  {'Withdrawable (profitable survivors):':<40} ${portfolio_profitable_pnl:,.2f}")
print(f"  {'Strategy total (all accounts):':<40} ${strategy_total_pnl:,.2f}")
print(f"  {'Portfolio (alive accounts):':<40} ${portfolio_alive_pnl:,.2f}")

print("\nACCOUNT STATISTICS")
print("-" * 60)
print(f"  {'Started:':<35} {number_accounts_started}")
print(f"  {'Alive:':<35} {n_alive}  ({n_alive / number_accounts_started * 100:.1f}%)")
print(f"  {'Blown:':<35} {n_blown}")
print(f"  {'Hit freeze trigger:':<35} {n_frozen}")

print("\nCAPITAL METRICS")
print("-" * 60)
print(f"  {'Total capital deployed:':<35} ${total_capital_deployed:,.2f}")
print(f"  {'Return on capital (alive P&L):':<35} {(portfolio_profitable_pnl / total_capital_deployed * 100):.1f}%")

if not monthly_pnl.empty:
    print("\nSINGLE ACCOUNT (monthly)")
    print("-" * 60)
    print(f"  {'Total:':<35} ${monthly_pnl.sum():,.2f}")
    print(f"  {'Win rate:':<35} {(monthly_pnl > 0).sum() / len(monthly_pnl) * 100:.1f}%"
          f"  ({(monthly_pnl > 0).sum()}/{len(monthly_pnl)})")
    print(f"  {'Best:':<35} ${monthly_pnl.max():,.2f}")
    print(f"  {'Avg:':<35} ${monthly_pnl.mean():,.2f}")
    print(f"  {'Worst:':<35} ${monthly_pnl.min():,.2f}")

if not portfolio_pnl.empty:
    pm = port_daily.resample("M").sum()
    print("\nPORTFOLIO (monthly)")
    print("-" * 60)
    print(f"  {'Total:':<35} ${pm.sum():,.2f}")
    print(f"  {'Win rate:':<35} {(pm > 0).sum() / len(pm) * 100:.1f}%"
          f"  ({(pm > 0).sum()}/{len(pm)})")
    print(f"  {'Best:':<35} ${pm.max():,.2f}")
    print(f"  {'Avg:':<35} ${pm.mean():,.2f}")
    print(f"  {'Worst:':<35} ${pm.min():,.2f}")

print("\n" + "=" * 60)
print("NOTE: MAE assumed before MFE (conservative blowout test).")
print(f"Total capital deployed across {number_accounts_started} accounts: ${total_capital_deployed:,.2f}")

try:
    plt.show()
except KeyboardInterrupt:
    print("\nStopped by user.")