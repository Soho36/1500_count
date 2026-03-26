import pandas as pd
import time
import random


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
        usecols=[
            'ts_event', 'open', 'high', 'low',
            'close', 'volume', 'symbol'
        ]
    )
    print(f"Rows loaded: {len(df):,}")

    print("Converting timestamps...")
    df['ts_event'] = pd.to_datetime(
        df['ts_event'],
        format='%Y-%m-%dT%H:%M:%S.%fZ',
        utc=True
    )

    # Convert to Exchange timezone to align with session times
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

    print("Synthesizing ticks from OHLC...")

    def ohlc_to_ticks(df, spread):
        random.seed(42)
        tick_rows = []

        for row in df.itertuples(index=False):
            ts = row.ts_event
            o = float(row.open)
            h = float(row.high)
            l = float(row.low)
            c = float(row.close)
            vol = row.volume

            # Heuristic: visit the closer extreme first
            if random.random() < 0.5:
                path = [o, h, l, c]
            else:
                path = [o, l, h, c]

            # Distribute bar volume across 4 ticks
            base_vol = vol // 4
            remainder = vol % 4

            for i, price in enumerate(path):
                tick_time = ts + pd.Timedelta(milliseconds=i * 200)

                bid = round(price - spread / 2, 2)
                ask = round(price + spread / 2, 2)
                last = round(price, 2)

                tick_vol = base_vol + (1 if i < remainder else 0)

                tick_rows.append([
                    tick_time.strftime('%Y.%m.%d'),
                    tick_time.strftime('%H:%M:%S.%f')[:-3],
                    bid,
                    ask,
                    last,
                    tick_vol
                ])

        tick_df = pd.DataFrame(
            tick_rows,
            columns=['<DATE>', '<TIME>', '<BID>', '<ASK>', '<LAST>', '<VOLUME>']
        )
        return tick_df

    output_df = ohlc_to_ticks(df, spread)

    print("Writing output file...")
    output_df.to_csv(
        output_file,
        index=False,
        float_format='%.2f'
    )

    end_time = time.perf_counter()

    print("\n✔ Tick file created successfully")
    print(f"Tick rows: {len(output_df):,}")
    print("Date range:", output_df['<DATE>'].min(), "→", output_df['<DATE>'].max())
    print(f"⏱ Total time: {end_time - start_time:.2f} seconds")

    return output_df


if __name__ == "__main__":
    folder_path = "F:\\DATABENTO\\mnq_last_days\\"
    in_file = f"{folder_path}mnq_ohlcv-1s.csv.zst"
    input_filename = in_file.split("\\")[-1].split(".")[0]  # Fixed extraction
    out_file = f"{folder_path}MT5_{input_filename}_ticks.csv"
    build_daily_roll_continuous(
        input_file=in_file,
        output_file=out_file,
        spread=0.25  # Adjust to your instrument's tick size
    )
