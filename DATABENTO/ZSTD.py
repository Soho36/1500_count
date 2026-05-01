import pandas as pd

print("Reading ZST file...")
df = pd.read_csv(r"F:\DATABENTO\ES\ES_full.ohlcv-1m.csv.zst")
print(df.head())
df_head = df.head()

print("Saving to CSV...")
df.to_csv(r"F:\DATABENTO\ES\ES_full.ohlcv-1m.csv", index=False)
print("Done")
