import pandas as pd
import matplotlib.pyplot as plt

# ======================
#  CONFIG
# ======================
# CSV_PATH = "CSVS/all_times_14_flat_ONLY_PNL.csv"
CSV_PATH = "CSVS/premarket_only.csv"
START_CAPITAL = 5000
NUM_ACCOUNTS = 1

# START_DATE = "2020-02-24"
# END_DATE = "2021-06-18"

START_DATE = None
END_DATE = None

USE_MANUAL_OFFSETS = True
MANUAL_OFFSETS = [360, 60, 90, 120, 15]  # in trading days

# ======================
#  FUNCTIONS
# ======================


def calculate_max_drawdown(equity):
    """Calculate maximum drawdown of an equity series."""
    rolling_max = equity.cummax()
    dd = equity - rolling_max
    return dd.min()


def build_equity_from_pl(pl_series, offset, starting_capital=5000):
    """Start the equity curve from <offset> days later using correct P&L values."""
    shifted_pl = pl_series.iloc[offset:]  # <-- FIXED
    equity = starting_capital + shifted_pl.cumsum()
    return equity


def compute_drawdown_series(equity):
    """Return drawdown series indexed by date."""
    rolling_max = equity.cummax()
    dd_series = equity - rolling_max
    return dd_series


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

# ======================
#  SELECT OFFSETS
# ======================
if USE_MANUAL_OFFSETS:
    best_offsets = MANUAL_OFFSETS[:NUM_ACCOUNTS]
    print(f"\n🔥 Using manual offsets: {best_offsets}")
else:
    print("Automatic offset search not implemented yet.")
    best_offsets = [0]

# ======================
#  PORTFOLIO MAX DD
# ======================
combined = None
for off in best_offsets:
    eq = build_equity_from_pl(pl, off, START_CAPITAL)
    if combined is None:
        combined = eq
    else:
        combined = combined.add(eq, fill_value=START_CAPITAL)

portfolio_dd = calculate_max_drawdown(combined)
print(f"\nPortfolio Max Drawdown = {portfolio_dd:.2f}")

# ======================
#  PLOTTING
# ======================

plt.figure(figsize=(14, 6))
plt.plot(equity_original.index, equity_original.values, label="Original", linewidth=2)
for off in best_offsets:
    eq = build_equity_from_pl(pl, off, START_CAPITAL)
    plt.plot(eq.index, eq.values, label=f"Offset {off} days")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ======================
#  DRAWDOWN PLOT
# ======================
dd_series = compute_drawdown_series(equity_original)

plt.figure(figsize=(14, 5))
plt.fill_between(dd_series.index, dd_series.values, 0, step="mid", color="blue")
plt.title("Drawdown Curve")
plt.ylabel("Drawdown")
plt.grid(True)
plt.tight_layout()

plt.show()
