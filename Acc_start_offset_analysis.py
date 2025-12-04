import pandas as pd
import matplotlib.pyplot as plt

# ======================
#  CONFIG
# ======================
CSV_PATH = "CSVS/all_times_14_flat.csv"
NUM_ACCOUNTS = 2
MAX_OFFSET = 40

START_DATE = "2021-01-12"               # set to None to disable filtering "YYYY-MM-DD"
END_DATE = "2021-12-29"                 # set to None to disable filtering "YYYY-MM-DD"
# START_DATE = None
# END_DATE = None


USE_MANUAL_OFFSETS = True               # set to True to use MANUAL_OFFSETS instead of calculating best offsets
MANUAL_OFFSETS = [30, 60, 90, 120, 150]   # used only if USE_MANUAL_OFFSETS = True

# ======================
#  FUNCTIONS
# ======================


def calculate_max_drawdown(equity):
    rolling_max = equity.cummax()
    dd = equity - rolling_max
    return dd.min()


def shift_equity(equity, offset):
    shifted = equity.shift(offset)
    # keep NaN — so curve is blank before start
    return shifted


# ======================
#  LOAD DATA
# ======================


df = pd.read_csv(CSV_PATH, sep="\t")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date")

# Force numeric cleanup


def clean_numeric(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    x = x.replace(" ", "")       # remove spaces like 1 234
    x = x.replace(",", ".")      # convert comma decimals
    x = x.replace("€", "")       # remove euro signs if any
    return x


for col in ["Quantity of positions", "Volume", "P.L", "Average P.L",
            "Comission", "Gross", "Net", "Open", "Hi", "Low", "Close"]:
    df[col] = df[col].apply(clean_numeric)
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Net"])

# --- Optional date filter ---
if START_DATE or END_DATE:
    if START_DATE:
        df = df[df["Date"] >= pd.to_datetime(START_DATE)]
    if END_DATE:
        df = df[df["Date"] <= pd.to_datetime(END_DATE)]
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"📅 Data filtered from {START_DATE or 'beginning'} to {END_DATE or 'end'}")


# ==============

df["Equity"] = df["Net"]
equity = df.set_index("Date")["Equity"]

print("Date range:", df["Date"].min(), "→", df["Date"].max())
print("Total rows after cleaning:", len(df))

# ======================
#  INVESTIGATE OFFSETS
# ======================

# ======================
#  INVESTIGATE OFFSETS
# ======================

offset_stats = []

for offset in range(MAX_OFFSET + 1):
    shifted = shift_equity(equity, offset)
    corr = equity.corr(shifted)
    max_dd = calculate_max_drawdown(shifted)

    offset_stats.append({
        "Offset": offset,
        "Correlation": corr,
        "MaxDD": max_dd
    })

offset_df = pd.DataFrame(offset_stats)
offset_df_sorted = offset_df.sort_values("Correlation")

# ---- Pick offsets ----
if USE_MANUAL_OFFSETS:
    best_offsets = MANUAL_OFFSETS[:NUM_ACCOUNTS]
    print(f"\n🔥 Using manual offsets: {best_offsets}")
else:
    best_offsets = offset_df_sorted["Offset"].head(NUM_ACCOUNTS).values.tolist()
    print("\n=== Best offsets (lowest correlation) ===")
    print(offset_df_sorted.head(10))
    print(f"\nSuggested offsets for {NUM_ACCOUNTS} accounts: {best_offsets}")



# ======================
#  COMPUTE PORTFOLIO MAX DD
# ======================

combined = None
for off in best_offsets:
    shifted = shift_equity(equity, off)
    combined = shifted if combined is None else combined + shifted

portfolio_max_dd = calculate_max_drawdown(combined)
print(f"\nPortfolio max DD = {portfolio_max_dd:.2f}")


# =====================================================
#  PLOTTING SECTION (3 PLOTS)
# =====================================================

# -----------------------------------------------------
# 1) Natural cycle plot
# -----------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(equity.index, equity.values, label="Equity Curve")
plt.title("Natural Cycle of Original Equity Curve")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()

# -----------------------------------------------------
# 2) Plot shifted curves for selected offsets
# -----------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(equity.index, equity.values, label="Original", linewidth=2)

for off in best_offsets:
    shifted = shift_equity(equity, off)
    plt.plot(shifted.index, shifted.values, label=f"Offset {off}")

plt.title("Shifted Equity Curves")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()

# -----------------------------------------------------
# 3) Calendar/timeline visualization of start days
# -----------------------------------------------------
plt.figure(figsize=(10, 4))

for idx, off in enumerate(best_offsets):
    plt.scatter([off], [idx], s=200)
    plt.text(off + 0.5, idx, f"Start +{off} days", va="center")

plt.yticks(range(len(best_offsets)), [f"Account {i+1}" for i in range(len(best_offsets))])
plt.xlabel("Day Offset")
plt.title("Staggered Start Timeline")
plt.grid(True, axis="x")
plt.tight_layout()
plt.show()
