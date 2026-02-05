import pandas as pd
# from datetime import datetime


def merge_trade_files_robust(file1_path, file2_path, output_path):
    """
    More robust version that handles potential column differences
    """
    # Read files, stripping whitespace from column names
    df1 = pd.read_csv(file1_path, sep='\t', encoding='utf-8')
    df2 = pd.read_csv(file2_path, sep='\t', encoding='utf-8')

    # Clean column names (remove trailing/leading whitespace)
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # Check for any column differences
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    if cols1 != cols2:
        print(f"Column differences detected!")
        print(f"Columns only in File 1: {cols1 - cols2}")
        print(f"Columns only in File 2: {cols2 - cols1}")

        # Use columns from the first file as reference
        common_cols = list(cols1.intersection(cols2))
        print(f"Using common columns: {common_cols}")

        # Select only common columns
        df1 = df1[common_cols]
        df2 = df2[common_cols]

    # Concatenate
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Convert Entry_time to datetime, try multiple formats
    try:
        merged_df['Entry_time'] = pd.to_datetime(merged_df['Entry_time'], format='%Y.%m.%d %H:%M:%S')
    except:
        try:
            merged_df['Entry_time'] = pd.to_datetime(merged_df['Entry_time'])
        except:
            print("Warning: Could not parse Entry_time as datetime")

    # Sort by Entry_time
    if 'Entry_time' in merged_df.columns:
        merged_df = merged_df.sort_values('Entry_time')

    # Save to file
    merged_df.to_csv(output_path, sep='\t', encoding='utf-8', index=False)

    print(f"Successfully merged files. Total trades: {len(merged_df)}")
    return merged_df


if __name__ == "__main__":
    # For direct script usage
    merge_trade_files_robust(
        "trade_stats_long_premarket.csv",
        "trade_stats_short_premarket.csv",
        "longs_shorts_merged_trades_premarket.csv"
    )
