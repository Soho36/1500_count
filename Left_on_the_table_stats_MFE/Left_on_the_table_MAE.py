import pandas as pd

# ==============================
# CONFIG
# ==============================
pd.set_option('display.max_rows', None)      # Show all rows when printing DataFrames
path = r"C:\Users\Vova deduskin lap\AppData\Roaming\MetaQuotes\Tester\870072DB5DBAB61841BAE146AFAAFB8A\Agent-127.0.0.1-3000\MQL5\Files\trade_stats.csv"

print("\n📥 Reading CSV file...")
df = pd.read_csv(path, sep="\t", encoding="utf-8")
print(f"✅ Loaded {len(df)} trades")

# ==============================
# CORE METRICS
# ==============================
print("\n📐 Computing risk and excursion metrics...")

# Risk in price units
df["R"] = df["MAE"].abs()

# Raw and normalized leftover
df["Left_raw"] = df["MFE"] - df["PNL"]
df["Left_R"] = df["Left_raw"] / df["R"]

# ==============================
# FEASIBLE TRADES (≥ 1R reached)
# ==============================
df_feasible = df[df["MFE"] >= df["R"]].copy()

print("\n📊 BASIC SUMMARY (only trades reaching ≥1R)")

summary = {
    "Total trades": f"{len(df)} - Total trades in the dataset",
    "Trades reaching ≥1R": f"{len(df_feasible)} - How many ever reached ≥1R (MFE ≥ |MAE|)",
    "Median Left_R": f"{df_feasible['Left_R'].median():.4f} - For 50% of feasible trades, left X on the table",
    "75th percentile Left_R": f"{df_feasible['Left_R'].quantile(0.75):.4f} - For 75% of feasible trades, left X on the table",
    "Mean Left_R (diagnostic)": f"{df_feasible['Left_R'].mean():.4f}",
}

for k, v in summary.items():
    print(f"{k:30s}: {v}")

sample_feasible = df_feasible[["MAE", "MFE", "PNL", "Left_R"]].sample(200, random_state=42)
print(f"\nFeasible sample of trades:\n {sample_feasible}")
# ==============================
# RISK SIZE BUCKET ANALYSIS
# ==============================
print("\n📦 Left_R by absolute MAE size (price buckets)")

df_feasible["MAE_price_bucket"] = pd.cut(
    df_feasible["R"],
    bins=[0, 10, 20, 40, 80, 200]
)

print(
    df_feasible
    .groupby("MAE_price_bucket")["Left_R"]
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
    "Median MAE in R": f"{(left['MAE'].abs() / left['R']).median()} - For trades where you could have made more, price first went almost fully to your stop",
    "Median Left_R": f"{left['Left_R'].median()}",
    "75th percentile Left_R": f"{left['Left_R'].quantile(0.75)}",
    "Mean Left_R": f"{left['Left_R'].mean()}",
}

for k, v in conditional.items():
    print(f"{k:30s}: {v}")

# ==============================
# MAE IN R BUCKETS
# ==============================
print("\n📉 Left_R by MAE severity (normalized MAE buckets) - \n"
      "If Zeroes - there are no trades where: MAE was small and profit was left on the table\n"
      "When MAE < 0.75R → you already extract what’s available:")

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

sample = left[["MAE", "MFE", "PNL", "Close_R", "Left_R"]].sample(20, random_state=42)
print(sample)

print("\n✅ Analysis complete.")
