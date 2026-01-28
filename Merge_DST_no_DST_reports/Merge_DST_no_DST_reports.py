import pandas as pd
import os


def merge_dst_and_nodst_csv_files(nodst_file, dst_file, output_file):
    """
    Merge DST and non-DST CSV files based on Entry_time column.

    Args:
        nodst_file (str): Path to file with non-DST dates
        dst_file (str): Path to file with DST dates
        output_file (str): Path for output merged file
    """

    # Read both CSV files with tab separator
    print(f"Reading {nodst_file}...")
    df_nodst = pd.read_csv(nodst_file, sep="\t", encoding="utf-8")

    print(f"Reading {dst_file}...")
    df_dst = pd.read_csv(dst_file, sep="\t", encoding="utf-8")

    print(f"Columns in nodst file: {list(df_nodst.columns)}")
    print(f"Columns in dst file: {list(df_dst.columns)}")

    # Ensure Entry_time column is datetime type (based on your CSV structure)
    # Using Entry_time for sorting since that's when the trade started
    if 'Entry_time' in df_nodst.columns:
        df_nodst['Entry_time'] = pd.to_datetime(df_nodst['Entry_time'])
        df_dst['Entry_time'] = pd.to_datetime(df_dst['Entry_time'])

        # Also convert Exit_time if present
        if 'Exit_time' in df_nodst.columns:
            df_nodst['Exit_time'] = pd.to_datetime(df_nodst['Exit_time'])
            df_dst['Exit_time'] = pd.to_datetime(df_dst['Exit_time'])

        # Combine both dataframes
        print("Combining dataframes...")
        df_combined = pd.concat([df_nodst, df_dst], ignore_index=True)

        # Sort by Entry_time to get continuous trades in chronological order
        df_combined = df_combined.sort_values('Entry_time').reset_index(drop=True)

        # Optional: Also sort by Ticket if you want trades in ticket order
        # df_combined = df_combined.sort_values(['Entry_time', 'Ticket']).reset_index(drop=True)

        # Reorder columns to match original structure
        original_columns = list(df_nodst.columns) if len(df_nodst) > 0 else list(df_dst.columns)
        df_combined = df_combined[original_columns]

        # Save to CSV
        print(f"Saving merged file to {output_file}...")
        df_combined.to_csv(output_file, index=False, sep="\t", encoding="utf-8")

        # Print summary
        print(f"\nSummary:")
        print(f"  Non-DST file: {len(df_nodst)} rows")
        print(f"  DST file: {len(df_dst)} rows")
        print(f"  Merged file: {len(df_combined)} rows")
        print(f"  Time range: {df_combined['Entry_time'].min()} to {df_combined['Entry_time'].max()}")

        return df_combined
    else:
        print("Error: 'Entry_time' column not found in the CSV files!")
        print(f"Available columns: {list(df_nodst.columns)}")
        return None


# Example usage
if __name__ == "__main__":
    # Define file paths - change these to match your actual file names
    nodst_file = "nodst.csv"  # Your file with non-DST dates
    dst_file = "dst.csv"  # Your file with DST dates
    output_file = "../MAE/only_good_windows_dst_nodst_premarket.csv"  # Output file

    # Check if files exist
    if not os.path.exists(nodst_file):
        print(f"Error: {nodst_file} not found!")
    elif not os.path.exists(dst_file):
        print(f"Error: {dst_file} not found!")
    else:
        # Merge the files
        merged_df = merge_dst_and_nodst_csv_files(nodst_file, dst_file, output_file)
        if merged_df is not None:
            print("\nMerge completed successfully!")

            # Optional: Display first few rows of merged data
            print("\nFirst 10 rows of merged data:")
            print(merged_df.head(10))

            # Optional: Show info about the dataframe
            print("\nDataFrame Info:")
            print(merged_df.info())