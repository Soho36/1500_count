import pandas as pd

print("Loading CSV...")
df = pd.read_csv(
    r"F:\DATABENTO\1s\MNQ_ohlcv-1s.csv",
    usecols=["ts_event", "symbol", "volume"]
)
print(f"Rows loaded: {len(df):,}")

print("Converting timestamps...")
df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)

print("Converting to timezone (America/Chicago)...")
df['ts_event'] = df['ts_event'].dt.tz_convert('America/Chicago')
df['ts_event'] = df['ts_event'].dt.tz_localize(None)

# Shift forward by 8h to put session close at 23:59 to match MT5's charts
df['ts_event'] = df['ts_event'] + pd.Timedelta(hours=8)

print("Extracting dates...")
df["trade_date"] = df["ts_event"].dt.date


print("Calculating dominant contract per day...")
daily = (
    df.groupby(["trade_date", "symbol"])["volume"]
      .sum()
      .reset_index()
)
print("Finding dominant contracts...")
dominant = (
    daily.sort_values(["trade_date", "volume"], ascending=[True, False])
         .drop_duplicates("trade_date")
)
print("Saving dominant contracts to CSV...")
dominant.to_csv(r"F:\DATABENTO\1s\dominant_contracts.csv", index=False)
print("Done.")
