import pandas as pd

print("Reading ZST file...")
df = pd.read_csv(r"C:\Source\DATABENTO\MNQ\MNQ_ohlcv-1s.csv.zst")
print(df.head())
df_head = df.head()

print("Saving to CSV...")
df.to_csv(r"C:\Source\DATABENTO\MNQ\MNQ_ohlcv-1s.csv", index=False)
print("Done")
