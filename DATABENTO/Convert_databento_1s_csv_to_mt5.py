import pandas as pd
import time


def build_daily_roll_continuous(
    input_file,
    output_file,
    spread
):
    start_time = time.perf_counter()
    print("Starting conversion...")

    dtypes = {
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'volume': 'int32',
        'symbol': 'category'
    }

    print("Loading CSV...")
    df = pd.read_csv(
        input_file,
        dtype=dtypes,
        usecols=['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol']
    )
    print(f"Rows loaded: {len(df):,}")

    print("Converting timestamps...")
    df['ts_event'] = pd.to_datetime(
        df['ts_event'],
        format='%Y-%m-%dT%H:%M:%S.%fZ',
        utc=True
    )

    print("Converting to timezone (America/Chicago)...")
    df['ts_event'] = df['ts_event'].dt.tz_convert('America/Chicago')
    df['ts_event'] = df['ts_event'].dt.tz_localize(None)

    # Shift forward by 8h to put session close at 23:59 to match MT5's charts
    df['ts_event'] = df['ts_event'] + pd.Timedelta(hours=8)

    print("Extracting dates...")
    df['trade_date'] = df['ts_event'].dt.date

    print("Calculating dominant contract per day...")
    daily_volume = (
        df.groupby(['trade_date', 'symbol'], observed=True)['volume']
          .sum()
          .reset_index()
    )
    dominant = (
        daily_volume.sort_values(['trade_date', 'volume'], ascending=[True, False])
                    .drop_duplicates('trade_date')
    )

    print("Merging dominant contracts...")
    df = df.merge(
        dominant[['trade_date', 'symbol']],
        on=['trade_date', 'symbol']
    )

    print("Sorting dataset...")
    df = df.sort_values('ts_event').reset_index(drop=True)

    print("Expanding 1s bars → 4 ticks each (open/high/low/close)...")

    n = len(df)
    date_str = df['ts_event'].dt.strftime('%Y.%m.%d')
    base_time = df['ts_event'].dt.strftime('%H:%M:%S')

    prices = {
        'open':  df['open'].values,
        'high':  df['high'].values,
        'low':   df['low'].values,
        'close': df['close'].values,
    }
    volume = df['volume'].values

    rows = []
    offsets = ['000', '025', '050', '075']
    ohlc_keys = ['open', 'high', 'low', 'close']
    vol_map = [0, 0, 0, 1]

    for i, offset, key, use_vol in zip(range(4), offsets, ohlc_keys, vol_map):
        price_col = prices[key]
        vol_col = volume if use_vol else [0] * n
        rows.append(pd.DataFrame({
            '<DATE>':    date_str,
            '<TIME>':    base_time + '.' + offset,
            '<BID>':     (price_col - spread).round(2),
            '<ASK>':     (price_col + spread).round(2),
            '<LAST>':    price_col.round(2),
            '<VOLUME>':  vol_col,
            '_sort_key': df['ts_event'].values.astype('int64') + i
        }))

    output_df = pd.concat(rows, ignore_index=True)
    output_df = output_df.sort_values('_sort_key').drop(columns='_sort_key').reset_index(drop=True)

    print("Writing output file...")
    output_df.to_csv(
        output_file,
        index=False,
        float_format='%.2f'
    )

    end_time = time.perf_counter()

    print("\n✔ Tick file created successfully")
    print(f"Tick rows: {len(output_df):,}  (= 4 per second bar: O/H/L/C)")
    print("Date range:", output_df['<DATE>'].min(), "→", output_df['<DATE>'].max())
    print(f"⏱ Total time: {end_time - start_time:.2f} seconds")

    return output_df


if __name__ == "__main__":
    folder_path = "C:\\Source\\DATABENTO\\MNQ\\todate"
    in_file = f"{folder_path}\\todate.csv"
    input_filename = in_file.split("\\")[-1].split(".")[0]
    out_file = f"MT5_{input_filename}_ticks.csv"
    build_daily_roll_continuous(
        input_file=in_file,
        output_file=out_file,
        spread=0.25
    )
