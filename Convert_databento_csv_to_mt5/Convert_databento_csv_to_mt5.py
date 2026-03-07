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

    # Convert to your local timezone (Estonia)
    print("Converting to local timezone (Europe/Tallinn)...")
    df['ts_event'] = df['ts_event'].dt.tz_convert('Europe/Tallinn')
    df['ts_event'] = df['ts_event'].dt.tz_localize(None)

    print("Extracting dates...")
    df['trade_date'] = df['ts_event'].dt.date

    print("Calculating dominant contract per day...")
    daily_volume = (
        df.groupby(['trade_date', 'symbol'])['volume']
          .sum()
          .reset_index()
    )

    dominant = (
        daily_volume.sort_values(['trade_date', 'volume'], ascending=[True, False])
                    .drop_duplicates('trade_date')
    )

    print("🔗 Merging dominant contracts...")
    df = df.merge(
        dominant[['trade_date', 'symbol']],
        on=['trade_date', 'symbol']
    )

    print("Sorting final dataset...")
    df = df.sort_values('ts_event')

    print("🛠 Formatting MT5 structure...")
    df['<DATE>'] = df['ts_event'].dt.strftime('%Y.%m.%d')
    df['<TIME>'] = df['ts_event'].dt.strftime('%H:%M:%S')

    df.rename(columns={
        'open': '<OPEN>',
        'high': '<HIGH>',
        'low': '<LOW>',
        'close': '<CLOSE>',
        'volume': '<TICKVOL>'
    }, inplace=True)

    df['<VOL>'] = df['<TICKVOL>']
    df['<SPREAD>'] = spread

    output_df = df[
        ['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>',
         '<TICKVOL>', '<VOL>', '<SPREAD>']
    ]

    print("Writing output file...")
    output_df.to_csv(
        output_file,
        sep='\t',
        index=False,
        float_format='%.2f'
    )

    end_time = time.perf_counter()

    print("\n✔ Continuous (daily-roll) file created")
    print(f"Rows: {len(output_df):,}")
    print("Date range:", output_df['<DATE>'].min(), "→", output_df['<DATE>'].max())
    print(f"⏱ Total time: {end_time - start_time:.2f} seconds")

    return output_df


if __name__ == "__main__":
    build_daily_roll_continuous(
        input_file="F:\\Databento\\mnq_last_days\\glbx-mdp3-20260227-20260305.ohlcv-1m.csv",
        output_file="F:\\Databento\\mnq_last_days\\mt5_last_days.csv",
        spread=1
    )
