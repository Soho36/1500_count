import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import time

# =================================================================================================
# CONFIG
# =================================================================================================

# Display number of rows in DataFrame outputs
pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', 20)

# Enable time filtering
ENTRY_TIME_FILTER = False

ENTRY_START_TIME = time(1, 0)
ENTRY_END_TIME = time(23, 30)

SAVE_FILES = False   # save output CSV files

SURVIVAL_CURVE_PLOT = False  # plot survival curve chart
HAZARD_RATE_PLOT = False  # plot hazard rate chart
EXPECTED_REMAINING_DD_PLOT = False  # plot expected remaining DD duration chart


# =================================================================================================
#  LOAD & CLEAN DATA
# =================================================================================================
try:
    input_path = "only_good_windows_dst_nodst_premarket.csv"  # input file from MT5 strategy tester
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

if ENTRY_TIME_FILTER:
    print(f"Applying time filter: {ENTRY_START_TIME} to {ENTRY_END_TIME}")
    df = df[
        (df["Entry_clock"] >= ENTRY_START_TIME) &
        (df["Entry_clock"] < ENTRY_END_TIME)
        ].copy()
    print(f"Filtered by time DF: {df.head(1000)}")
else:
    print("No time filtering applied.")
    print(f"Full DF: {df}")


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
for n in [5, 10, 20, 30, 60, 90]:
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

# =====================================================================
# EXPECTED REMAINING DD DURATION
# =====================================================================

exp_rows = []

days = surv_df["Days"].values
S = surv_df["Survival_Prob"].values

for i, t in enumerate(days):
    if S[i] <= 0:
        continue

    # numerical integral of survival curve tail
    remaining_area = 0.0
    for j in range(i, len(days) - 1):
        dt = days[j + 1] - days[j]
        remaining_area += S[j] * dt

    expected_remaining = remaining_area / S[i]

    exp_rows.append({
        "Days_In_DD": t,
        "Expected_Remaining_Days": expected_remaining
    })

exp_df = pd.DataFrame(exp_rows)


# =====================================================================
# HAZARD RATE (Conditional recovery probability)
# =====================================================================

hazard_rows = []

timeline = np.sort(np.unique(dur_kaplan))

for t in timeline:
    at_risk = (dur_kaplan >= t).sum()
    recovered = ((dur_kaplan == t) & (events == 1)).sum()

    if at_risk > 0:
        hazard = recovered / at_risk
    else:
        hazard = np.nan

    hazard_rows.append({
        "Days": t,
        "At_Risk": at_risk,
        "Recovered": recovered,
        "Hazard": hazard
    })

hazard_df = pd.DataFrame(hazard_rows)

hazard_df["Support"] = hazard_df["At_Risk"]
hazard_clean = hazard_df[hazard_df["Support"] >= 10].copy()
hazard_clean["Hazard_SMA"] = hazard_clean["Hazard"].rolling(5, min_periods=3).mean()


# ============================================
# Drawdown survival curve (Kaplan-Meier survival estimation)
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

if SURVIVAL_CURVE_PLOT:
    plt.step(surv_df["Days"], surv_df["Survival_Prob"], where="post")
    plt.xlabel("Days in Drawdown")
    plt.ylabel("P(DD not recovered)")
    plt.title("Drawdown Survival Curve")
    plt.grid(True)


# ============================================
# Hazard rate plot
# ============================================
"""
Hazard rate at day t means:
If a drawdown has lasted at least "t" days, what is the probability it recovers on day "t"+1?

This answers:
"Do DDs heal faster or slower as time passes?"

How to interpret the hazard curve
Case A — Hazard decreases over time

Early DDs recover easily
Long DDs are sticky
Suggests structural tail risk
⚠️ Dangerous for accounts

Case B — Hazard flat
Memoryless DDs
Time doesn’t matter much
Pure variance-driven system
✅ Statistically clean

Case C — Hazard increases
Market eventually “fixes” DDs
Mean reversion in time
Rare but excellent
🚀 Very robust strategy
"""
if HAZARD_RATE_PLOT:
    plt.figure(figsize=(10, 4))
    plt.plot(hazard_clean["Days"], hazard_clean["Hazard"], marker="o")
    plt.xlabel("Days in Drawdown")
    plt.ylabel("P(Recovery tomorrow)")
    plt.title("Drawdown Recovery Hazard Rate")
    plt.grid(True)

# ============================================
# Expected remaining DD duration plot
# ============================================

if EXPECTED_REMAINING_DD_PLOT:
    plt.figure(figsize=(10, 4))
    plt.plot(
        exp_df["Days_In_DD"],
        exp_df["Expected_Remaining_Days"],
        linewidth=2
    )
    plt.xlabel("Days already in Drawdown")
    plt.ylabel("Expected remaining DD days")
    plt.title("Expected Remaining Drawdown Duration")
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
