import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================
pd.set_option('display.max_rows', None)      # Show all rows when printing DataFrames
# path = (r"C:\Users\Vova deduskin lap\AppData\Roaming\MetaQuotes\Tester\870072DB5DBAB61841BAE146AFAAFB8A"
#         r"\Agent-127.0.0.1-3000\MQL5\Files\trade_stats.csv")
path = (r"C:\Users\Liikurserv\AppData\Roaming\MetaQuotes\Tester\F5855995045EF8A4C3CA7AE968872CF2"
        r"\Agent-127.0.0.1-3000\MQL5\Files\trade_stats.csv")

print("\n📥 Reading CSV file...")
df = pd.read_csv(path, sep="\t", encoding="utf-8")
print(f"✅ Loaded {len(df)} trades")

# ==============================
# CORE METRICS
# ==============================
print("\n📐 Computing risk and excursion metrics...")

# Risk in price units
df["R"] = df["Stop_money"]
assert (df["R"] > 0).all(), "All R values must be positive"

# Normalized metrics
df["MAE_R"] = df["MAE"].abs() / df["R"]
df["Unrealized_R"] = df["MFE"] / df["R"]
df["Left_R"] = np.nan
mask = df["PNL"] > 0
df.loc[mask, "Left_R"] = (df.loc[mask, "MFE"] - df.loc[mask, "PNL"]) / df.loc[mask, "R"]

# ==============================
# FEASIBLE TRADES (≥ 1R reached)
# ==============================
df_feasible = df[df["MFE"] >= df["R"]].copy()

print("\n📊 BASIC SUMMARY (only trades reaching ≥1R)")

summary = {
    "Total trades": f"{len(df)} - Total trades in the dataset",
    "Trades reaching ≥1R": f"{len(df_feasible)} - How many ever reached ≥1R (MFE ≥ stop distance)",
    "Median Left_R": f"{df_feasible['Left_R'].median():.4f} - For 50% of winning trades that reached ≥1R, the maximum unrealized profit was this much left on the table",
    "75th percentile Left_R": f"{df_feasible['Left_R'].quantile(0.75):.4f} - For 75% of winning trades that reached ≥1R, the maximum unrealized profit was this much left on the table",
    "Mean Left_R (diagnostic)": f"{df_feasible['Left_R'].mean():.4f}",
}

for k, v in summary.items():
    print(f"{k:30s}: {v}")

sample_feasible = df_feasible[["Stop_money", "MAE", "MFE", "PNL", "Left_R"]].sample(20, random_state=42)
print(f"\nFeasible sample of trades:\n {sample_feasible}")
# ==============================
# RISK SIZE BUCKET ANALYSIS
# ==============================
print("\n📦 Left_R by stop size (price buckets). 0-10$, 10-2$, 20-40$, etc.")

df_feasible["Stop_size_bucket"] = pd.cut(
    df_feasible["R"],
    bins=[0, 10, 20, 40, 80, 200]
)

print(
    df_feasible
    .groupby("Stop_size_bucket")["Left_R"]
    .median()
)

# ==============================
# CONDITIONAL ANALYSIS:
# ONLY TRADES WHERE SOMETHING WAS LEFT
# ==============================
left = df_feasible[df_feasible["Left_R"] > 0].copy()
# print(left)
print("\n🔍 CONDITIONAL ANALYSIS (only trades with Left_R > 0)")

conditional = {
    "Count": f"{len(left)} -  How many trades reached ≥1R (MFE ≥ R) and left profit on the table",
    "Median MAE in R": f"{(left['MAE'].abs() / left['R']).median()}",
    "Median Left_R": f"{left['Left_R'].median()}",
    "75th percentile Left_R": f"{left['Left_R'].quantile(0.75)}",
}

for k, v in conditional.items():
    print(f"{k:30s}: {v}")

# ==============================
# MAE IN R BUCKETS
# ==============================

df_feasible["MAE_R_bin"] = pd.cut(
    df_feasible["MAE"].abs() / df_feasible["R"],
    bins=[0, 0.25, 0.5, 0.75, 1.0]
)

print(
    df_feasible
    .groupby("MAE_R_bin")["Left_R"]
    .median()
)

# ==============================
# CONCRETE TRADE INSPECTION
# ==============================
print("\n🧪 Random sample of trades where profit was left:")

left["Close_R"] = left["PNL"] / left["R"]

sample = left[["Stop_money", "MAE", "MFE", "PNL", "Close_R", "Left_R"]].sample(20, random_state=42)
print(sample)

print("\n✅ Analysis complete.")

print("\n💾 Preparing tables for Excel export...")

# ==============================
# PREPARE TABLES FOR EXCEL
# ==============================

# Summary table
summary_df = pd.DataFrame(
    {
        "Metric": summary.keys(),
        "Explanation & Value": summary.values()
    }
)

# Conditional summary table
conditional_df = pd.DataFrame(
    {
        "Metric": conditional.keys(),
        "Explanation & Value": conditional.values()
    }
)

# MAE price bucket table
mae_price_table = (
    df_feasible
    .groupby("Stop_size_bucket")["Left_R"]
    .median()
    .reset_index()
    .rename(columns={"Left_R": "Median_Left_R"})
)

# MAE normalized bucket table
mae_r_table = (
    df_feasible
    .groupby("MAE_R_bin")["Left_R"]
    .median()
    .reset_index()
    .rename(columns={"Left_R": "Median_Left_R"})
)

# All feasible trades
feasible_trades = df_feasible[
    ["Entry_time", "Exit_time", "MAE", "MFE", "PNL", "R", "Left_R"]
].copy()

# Feasible trades where profit was left
feasible_left_trades = left[
    ["Entry_time", "Exit_time", "MAE", "MFE", "PNL", "R", "Close_R", "Left_R"]
].copy()

stress_trades = df[
    (df["MFE"] >= df["R"]) &
    (df["PNL"] > 0) &
    (df["MAE_R"] >= 0.75) & (df["MAE_R"] <= 1.05)
]


print("\n🏁 All done!")

# ==============================
# EXPORT TO EXCEL
# ==============================

output_path = "left_on_table_analysis.xlsx"

print(f"\n📤 Exporting results to Excel: {output_path}")

with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    conditional_df.to_excel(writer, sheet_name="Conditional", index=False)
    mae_price_table.to_excel(writer, sheet_name="Stop_size_bucket", index=False)
    mae_r_table.to_excel(writer, sheet_name="MAE_R_buckets", index=False)
    feasible_left_trades.to_excel(writer, sheet_name="Feasible_Left", index=False)
    feasible_trades.to_excel(writer, sheet_name="Feasible_All", index=False)
    stress_trades.to_excel(writer, sheet_name="High_MAE_Stress_Trades", index=False)

print("✅ Excel export completed.")
