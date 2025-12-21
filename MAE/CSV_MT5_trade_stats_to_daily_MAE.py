import pandas as pd
import matplotlib.pyplot as plt

# ======================
#  LOAD & CLEAN DATA
# ======================
try:
    input_path = "trade_stats_1_minute.csv"  # input file from MT5 strategy tester
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

df["Date"] = df["Entry_time"].dt.date

# ======================
# MAE / MFE LOGIC
# ======================

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


# ======================
# PROP-FIRM TRAILING DD
# ======================

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


# Aggregate to daily
dd_curve["Date"] = dd_curve["time"].dt.date

daily = dd_curve.groupby("Date").agg(
    PNL=("equity", "last"),
    MAE=("equity", "min"),
    MFE=("equity", "max")
).reset_index()


# ======================
# SAVE OUTPUTS
# ======================

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


# noinspection PyTypeChecker
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# ======================
# 1) Equity curves
# ======================
axes[0].plot(plot_df["Date"], plot_df["Equity"], linewidth=2, label="Equity")
axes[0].plot(plot_df["Date"], plot_df["Equity_Peak"], linewidth=1, label="Equity_Peak")
axes[0].plot(plot_df["Date"], plot_df["Equity_Low"], linewidth=1, label="Equity_Low")
axes[0].set_title("Equity Curve")
axes[0].set_ylabel("PNL")
axes[0].grid(True)
axes[0].legend()

# ======================
# 2) Closed equity DD
# ======================
axes[1].plot(plot_df["Date"], plot_df["DD_Closed"], linewidth=2, label="Closed DD")
axes[1].axhline(0, linewidth=0.8)
axes[1].set_title("Closed Equity Drawdown")
axes[1].set_ylabel("PNL")
axes[1].grid(True)
axes[1].legend()

# ======================
# 3) Floating (prop-style) DD
# ======================
axes[2].plot(plot_df["Date"], plot_df["DD_Floating"], linewidth=2, label="Floating DD")
axes[2].axhline(0, linewidth=0.8)
axes[2].set_title("Floating Drawdown (Prop Style)")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("PNL")
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()
