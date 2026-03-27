import pandas as pd

df = pd.read_csv("F:\DATABENTO\mnq_last_days\MNQ_ohlcv-1s.csv.zst")
print(df.head())
df_head = df.head()

df.to_csv("F:\DATABENTO\mnq_last_days\MNQ_ohlcv-1s.csv", index=False)
print("Done")