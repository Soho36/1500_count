import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import time

# =================================================================================================
# CONFIG
# =================================================================================================

# Display number of rows in DataFrame outputs
pd.set_option('display.max_rows', 500)

# Filter to pre-market trades only (1:00 - 10:00)
PREMARKET_START = time(1, 0)
PREMARKET_END = time(10, 0)

SAVE_FILES = False   # save output CSV files


# =================================================================================================
#  LOAD & CLEAN DATA
# =================================================================================================
try:
    input_path = "../CSVS/time_shifted_trade_stats_1_minute.csv"  # input file from MT5 strategy tester
    df = pd.read_csv(input_path, sep="\t")

    for col in ["MAE", "MFE", "PNL"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(' ', '', regex=False)
                .str.replace(',', '.', regex=False)
                .astype(float)
            )

    df["Entry_time"] = pd.to_datetime(df["Entry_time"], format="mixed", dayfirst=True)
    df["Exit_time"] = pd.to_datetime(df["Exit_time"], format="mixed", dayfirst=True)

    df = df[["Entry_time", "Exit_time", "MAE", "MFE", "PNL"]].dropna()
except FileNotFoundError:
    print("Input file not found. Please ensure 'MAE/trade_stats_1_minute.csv' exists.")
    exit()

# ======================
df["Date"] = df["Entry_time"].dt.date

# extract time
df["Entry_clock"] = df["Entry_time"].dt.time

df = df[
    (df["Entry_clock"] >= PREMARKET_START) &
    (df["Entry_clock"] < PREMARKET_END)
].copy()

print(f"Filtered by time DF: {df}")

# =================================================================================================
# MAE / MFE LOGIC
# =================================================================================================

rows = []
equity = 0.0

for _, r in df.sort_values("Entry_time").iterrows():

    # entry
    rows.append({
        "time": r["Entry_time"],
        "equity": equity,
        "event": "entry"
    })

    # worst excursion
    rows.append({
        "time": r["Entry_time"],   # exact time unknown
        "equity": equity + r["MAE"],
        "event": "mae"
    })

    # best excursion
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


# =================================================================================================
# PROP-FIRM TRAILING DD
# =================================================================================================

equity_close = 0.0      # cumulative closed PNL
equity_peak = 0.0       # highest floating equity ever
worst_dd = 0.0          # worst trailing DD observed

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

# Prepare daily plot data
plot_df = (
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

plot_df["Date"] = pd.to_datetime(plot_df["Date"])
#  Closed DD calculation
plot_df["Closed_Peak"] = plot_df["Equity"].cummax()
plot_df["DD_Closed"] = plot_df["Equity"] - plot_df["Closed_Peak"]

#   MAX DD STATISTICS
print("Max closed DD:", plot_df["DD_Closed"].min())
print("Max floating DD:", plot_df["DD_Floating"].min())
print("Final PNL:", plot_df["Equity"].iloc[-1])

# Aggregate to daily
dd_curve["Date"] = dd_curve["time"].dt.date


# =================================================================================================
# DD duration statistics
# =================================================================================================

dd = plot_df[["Date", "DD_Closed"]].copy()
dd = dd.sort_values("Date").reset_index(drop=True)

dd["In_DD"] = dd["DD_Closed"] < 0

dd_periods = []

dd_start = None

for i, r in dd.iterrows():
    if r["In_DD"] and dd_start is None:
        # DD starts
        dd_start = r["Date"]

    elif not r["In_DD"] and dd_start is not None:
        # DD ends
        dd_end = r["Date"]
        duration = (dd_end - dd_start).days + 1

        dd_periods.append({
            "DD_Start": dd_start,
            "DD_End": dd_end,
            "Duration_Days": duration
        })

        dd_start = None

# handle open DD at the end
if dd_start is not None:
    dd_periods.append({
        "DD_Start": dd_start,
        "DD_End": None,
        "Duration_Days": (dd["Date"].iloc[-1] - dd_start).days,
        "Open_DD": True
    })

dd_stats = pd.DataFrame(dd_periods)

closed_dds = dd_stats[dd_stats["DD_End"].notna()]   # Exclude open DDs from duration stats (avoid mean inflation)
dur = closed_dds["Duration_Days"]

print("\nDrawdown duration statistics (days):")
print(f"Count DDs: {len(dur)}")
print(f"Mean DD duration:   {dur.mean():.2f}")
print(f"Median DD duration: {dur.median():.2f}")
print(f"Max DD duration:    {dur.max():.0f}\n")
for n in [5, 10, 20]:
    pct = (dur > n).mean() * 100
    print(f"% of DDs longer than {n} days: {pct:.1f}%")

# =================================================================================================
# KAPLAN–MEIER SURVIVAL ESTIMATE FOR DD DURATIONS
# =================================================================================================

dur_kaplan = dd_stats["Duration_Days"].values
events = dd_stats["DD_End"].notna().astype(int).values
# event = 1 → recovered, 0 → still open (censored)

# Kaplan–Meier
timeline = np.arange(1, dur_kaplan.max() + 1)
survival = []

S = 1.0
for t in timeline:
    at_risk = (dur_kaplan >= t).sum()
    recovered = ((dur_kaplan == t) & (events == 1)).sum()

    if at_risk > 0:
        S *= (1 - recovered / at_risk)

    survival.append((t, S))

surv_df = pd.DataFrame(survival, columns=["Days", "Survival_Prob"])

# ============================================
# 4) Drawdown survival curve (Kaplan-Meier survival estimation)
# ============================================

"""
How to interpret it (key intuition)
Example interpretation:
Day	Survival S(t)
5	0.60
10	0.30
20	0.08

Meaning:
60% of DDs last longer than 5 days
Only 8% last longer than 20 days

So if your current DD = 20 days →
You’re in the worst 8% historically.
"""

plt.step(surv_df["Days"], surv_df["Survival_Prob"], where="post")
plt.xlabel("Days in Drawdown")
plt.ylabel("P(DD not recovered)")
plt.title("Drawdown Survival Curve")
plt.grid(True)


# ============================================
# 1) Equity curves
# ============================================
# noinspection PyTypeChecker
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
axes[0].plot(plot_df["Date"], plot_df["Equity"], linewidth=2, label="Equity")
axes[0].plot(plot_df["Date"], plot_df["Equity_Peak"], linewidth=1, label="Equity_Peak")
axes[0].plot(plot_df["Date"], plot_df["Equity_Low"], linewidth=1, label="Equity_Low")
axes[0].set_title("Equity Curve")
axes[0].set_ylabel("PNL")
axes[0].grid(True)
axes[0].legend()

# ============================================
# 2) Closed equity DD
# ============================================
axes[1].plot(plot_df["Date"], plot_df["DD_Closed"], linewidth=2, label="Closed DD")
axes[1].axhline(0, linewidth=0.8)
axes[1].set_title("Closed Equity Drawdown")
axes[1].set_ylabel("PNL")
axes[1].grid(True)
axes[1].legend()

# ============================================
# 3) Floating (prop-style) DD
# ============================================
axes[2].plot(plot_df["Date"], plot_df["DD_Floating"], linewidth=2, label="Floating DD")
axes[2].axhline(0, linewidth=0.8)
axes[2].set_title("Floating Drawdown (Prop Style)")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("PNL")
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()

try:
    plt.show()
except KeyboardInterrupt as e:
    print(f"Script stopped by user: {e}")

# =================================================================================================
# SAVE OUTPUTS
# =================================================================================================
daily = dd_curve.groupby("Date").agg(
    PNL=("equity", "last"),
    MAE=("equity", "min"),
    MFE=("equity", "max")
).reset_index()

if SAVE_FILES:
    daily_output_mae = "../MAE/daily_results_mae.csv"
    daily_output_mae_equity_peak_low = "../MAE/daily_results_mae_eq_peak_low.csv"

    try:
        (daily.assign(Date=daily["Date"].apply(lambda d: d.strftime("%d.%m.%Y")))
         .to_csv(daily_output_mae, index=False, sep="\t", encoding="utf-8"))
        print(f"Daily MAE/MFE saved to: {daily_output_mae}")
    except Exception as e:
        print(f"Error saving to {daily_output_mae}: {e}")

    try:
        dd_curve.to_csv(daily_output_mae_equity_peak_low, index=False, sep="\t", encoding="utf-8")
        print(f"Prop trailing DD saved to: {daily_output_mae_equity_peak_low}")
    except Exception as e:
        print(f"Error saving to {daily_output_mae_equity_peak_low}: {e}")
