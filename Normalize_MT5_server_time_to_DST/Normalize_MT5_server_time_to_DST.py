import pandas as pd
from datetime import datetime, timedelta
import pytz


def adjust_timestamp_to_usa_exchange(row_date_str, row_time_str):
    """
    Adjust broker timestamp to USA exchange time (EST/EDT).
    Assumes broker time is 2-3 hours behind USA exchange time.
    """
    # Parse the date and time
    dt_str = f"{row_date_str} {row_time_str}"
    broker_dt = datetime.strptime(dt_str, '%Y.%m.%d %H:%M:%S')

    # Create timezone objects
    # Broker timezone (assuming it's GMT-3 during DST and GMT-2 otherwise)
    # We'll treat broker time as a naive datetime and adjust it

    # USA exchange timezone (Eastern Time)
    eastern = pytz.timezone('US/Eastern')

    # Convert broker datetime to Eastern Time by adding the offset
    # We need to determine the correct offset based on the date

    # For USA Eastern Time:
    # - EST (winter): UTC-5
    # - EDT (summer): UTC-4
    # If broker is 2-3 hours behind, we need to add:
    # - 2 hours when USA is in EST (winter)
    # - 3 hours when USA is in EDT (summer)

    # First, create a datetime in Eastern timezone for this date
    # We'll use noon to avoid any edge cases around midnight
    eastern_dt_guess = eastern.localize(
        datetime(broker_dt.year, broker_dt.month, broker_dt.day, 12, 0, 0)
    )

    # Check if this datetime is in DST (EDT) or not (EST)
    is_dst = eastern_dt_guess.dst() != timedelta(0)

    # Apply the appropriate offset
    if is_dst:
        # USA in EDT (summer), broker is 3 hours behind
        adjusted_dt = broker_dt + timedelta(hours=3)
    else:
        # USA in EST (winter), broker is 2 hours behind
        adjusted_dt = broker_dt + timedelta(hours=2)

    # Convert to Eastern timezone to get proper DST info
    # First localize the adjusted datetime as naive, then convert to Eastern
    adjusted_dt_eastern = eastern.localize(adjusted_dt)

    return adjusted_dt_eastern


def process_csv_file(input_file, output_file):
    """
    Process the CSV file and adjust timestamps
    """
    # Read the CSV file
    df = pd.read_csv(input_file, sep='\t')

    print(f"Processing {len(df)} rows...")

    # Create new columns for adjusted timestamps
    adjusted_dates = []
    adjusted_times = []

    for index, row in df.iterrows():
        adjusted_dt = adjust_timestamp_to_usa_exchange(row['<DATE>'], row['<TIME>'])

        # Format back to original string formats
        adjusted_dates.append(adjusted_dt.strftime('%Y.%m.%d'))
        adjusted_times.append(adjusted_dt.strftime('%H:%M:%S'))

        # Progress update
        if index % 10000 == 0 and index > 0:
            print(f"Processed {index} rows...")

    # Create new DataFrame with adjusted timestamps
    df_adjusted = df.copy()
    df_adjusted['<DATE>'] = adjusted_dates
    df_adjusted['<TIME>'] = adjusted_times

    # Sort by new timestamp to ensure chronological order
    df_adjusted['DATETIME'] = pd.to_datetime(
        df_adjusted['<DATE>'] + ' ' + df_adjusted['<TIME>'],
        format='%Y.%m.%d %H:%M:%S'
    )
    df_adjusted = df_adjusted.sort_values('DATETIME')
    df_adjusted = df_adjusted.drop('DATETIME', axis=1)

    # Save to new file
    df_adjusted.to_csv(output_file, sep='\t', index=False)
    print(f"Saved adjusted data to {output_file}")

    # Display some samples
    print("\nSample of original data (first 3 rows):")
    print(df.head(3).to_string())
    print("\nSample of adjusted data (first 3 rows):")
    print(df_adjusted.head(3).to_string())


def main():
    # Configuration
    input_file = 'MNQ_merged_no_spread.csv'  # Change this to your input file name
    output_file = 'MNQ_merged_no_spread_DST_adjusted.csv'  # Output file name

    process_csv_file(input_file, output_file)

    print("\nDone! Data has been adjusted to USA exchange time (EST/EDT).")


if __name__ == "__main__":
    main()