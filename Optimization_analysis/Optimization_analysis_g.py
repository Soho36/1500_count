import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================

FILE_PATH = "0130 0200.csv"   # or .xlsx
MIN_PROFIT = 0
MIN_TRADES = 30
MAX_DD = 20
MIN_PF = 1.1
MIN_SHARPE = 5
MIN_ACTIVE_MONTHS = 2

# Composite score weights (must sum to 1)
WEIGHTS = {
    "Sharpe": 0.20,
    "PF": 0.25,
    "Recovery": 0.20,
    "ExpectedPayoff": 0.15,
    "DD": 0.10
}

MONTH_COLS = [
    "TradeJanuary", "TradeFebruary", "TradeMarch", "TradeApril",
    "TradeMay", "TradeJune", "TradeJuly", "TradeAugust",
    "TradeSeptember", "TradeOctober", "TradeNovember", "TradeDecember"
]

# =========================
# LOAD DATA
# =========================

if FILE_PATH.endswith(".csv"):
    df = pd.read_csv(FILE_PATH, sep='\t', encoding='utf-8')
else:
    df = pd.read_excel(FILE_PATH)

# =========================
# BASIC CLEANING
# =========================

# Normalize boolean month columns
df[MONTH_COLS] = df[MONTH_COLS].astype(bool)

# Active months count
df["ActiveMonths"] = df[MONTH_COLS].sum(axis=1)

# =========================
# HARD FILTERS
# =========================

filtered = df[
    (df["Profit"] > MIN_PROFIT) &
    (df["Trades"] >= MIN_TRADES) &
    (df["Equity DD %"] <= MAX_DD) &
    (df["Profit Factor"] >= MIN_PF) &
    (df["Sharpe Ratio"] >= MIN_SHARPE) &
    (df["ActiveMonths"] >= MIN_ACTIVE_MONTHS)
].copy()

if filtered.empty:
    raise ValueError("No runs passed hard filters — relax thresholds.")

# =========================
# PERCENTILE NORMALIZATION
# =========================

def pct_rank(series, higher_is_better=True):
    r = series.rank(pct=True)
    return r if higher_is_better else 1 - r

filtered["Sharpe_rank"] = pct_rank(filtered["Sharpe Ratio"], True)
filtered["PF_rank"] = pct_rank(filtered["Profit Factor"], True)
filtered["Recovery_rank"] = pct_rank(filtered["Recovery Factor"], True)
filtered["ExpectedPayoff_rank"] = pct_rank(filtered["Expected Payoff"], True)
filtered["DD_rank"] = pct_rank(filtered["Equity DD %"], False)

# =========================
# COMPOSITE SCORE
# =========================

filtered["Score"] = (
    WEIGHTS["Sharpe"] * filtered["Sharpe_rank"] +
    WEIGHTS["PF"] * filtered["PF_rank"] +
    WEIGHTS["Recovery"] * filtered["Recovery_rank"] +
    WEIGHTS["ExpectedPayoff"] * filtered["ExpectedPayoff_rank"] +
    WEIGHTS["DD"] * filtered["DD_rank"]
)

# =========================
# PARETO FRONT (PF↑, Sharpe↑, DD↓)
# =========================

def pareto_front(df, maximize_cols, minimize_cols):
    is_pareto = np.ones(len(df), dtype=bool)

    for i, row in df.iterrows():
        for j, other in df.iterrows():
            if i == j:
                continue

            better_or_equal = True
            strictly_better = False

            for c in maximize_cols:
                if other[c] < row[c]:
                    better_or_equal = False
                elif other[c] > row[c]:
                    strictly_better = True

            for c in minimize_cols:
                if other[c] > row[c]:
                    better_or_equal = False
                elif other[c] < row[c]:
                    strictly_better = True

            if better_or_equal and strictly_better:
                is_pareto[df.index.get_loc(i)] = False
                break

    return df[is_pareto]

pareto = pareto_front(
    filtered,
    maximize_cols=["Profit Factor", "Sharpe Ratio"],
    minimize_cols=["Equity DD %"]
)

# =========================
# FINAL SELECTION
# =========================

TOP_N = 10

final = (
    pareto
    .sort_values("Score", ascending=False)
    .head(TOP_N)
)

# =========================
# OUTPUT
# =========================

cols_to_show = [
    "Pass", "Score", "Profit", "Trades",
    "Profit Factor", "Sharpe Ratio",
    "Recovery Factor", "Equity DD %",
    "ActiveMonths"
]

print("\nTOP CANDIDATE RUNS:\n")
print(final[cols_to_show])

final.to_csv("selected_runs.csv", index=False)
