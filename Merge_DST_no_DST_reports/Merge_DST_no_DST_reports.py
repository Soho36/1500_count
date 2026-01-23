import pandas as pd
import os


def merge_dst_and_nodst_files(nodst_file, dst_file, output_file):
    """
    Merge DST and non-DST Excel files based on Time column.

    Args:
        nodst_file (str): Path to file with non-DST dates
        dst_file (str): Path to file with DST dates
        output_file (str): Path for output merged file
    """

    # Read both Excel files
    print(f"Reading {nodst_file}...")
    df_nodst = pd.read_csv(nodst_file, sep="\t", encoding="utf-8")

    print(f"Reading {dst_file}...")
    df_dst = pd.read_csv(dst_file, sep="\t", encoding="utf-8")

    # Ensure Time column is datetime type
    df_nodst['Time'] = pd.to_datetime(df_nodst['Time'])
    df_dst['Time'] = pd.to_datetime(df_dst['Time'])

    # Combine both dataframes
    print("Combining dataframes...")
    df_combined = pd.concat([df_nodst, df_dst], ignore_index=True)

    # Sort by Time to get continuous trades
    df_combined = df_combined.sort_values('Time').reset_index(drop=True)

    # Optional: Sort other columns if needed
    # You might want to reorder columns to match original structure
    original_columns = list(df_nodst.columns) if len(df_nodst) > 0 else list(df_dst.columns)
    df_combined = df_combined[original_columns]

    # Save to Excel
    print(f"Saving merged file to {output_file}...")
    df_combined.to_csv(output_file, index=False, sep="\t", encoding="utf-8")

    # Print summary
    print(f"\nSummary:")
    print(f"  Non-DST file: {len(df_nodst)} rows")
    print(f"  DST file: {len(df_dst)} rows")
    print(f"  Merged file: {len(df_combined)} rows")
    print(f"  Time range: {df_combined['Time'].min()} to {df_combined['Time'].max()}")

    return df_combined


# Example usage
if __name__ == "__main__":
    # Define file paths - change these to match your actual file names
    nodst_file = "nodst.csv"  # Your file with non-DST dates
    dst_file = "dst.csv"  # Your file with DST dates
    output_file = "../MAE/merged_trades_dst_plus_nodst.csv"  # Output file

    # Check if files exist
    if not os.path.exists(nodst_file):
        print(f"Error: {nodst_file} not found!")
    elif not os.path.exists(dst_file):
        print(f"Error: {dst_file} not found!")
    else:
        # Merge the files
        merged_df = merge_dst_and_nodst_files(nodst_file, dst_file, output_file)
        print("\nMerge completed successfully!")

        # Optional: Display first few rows of merged data
        print("\nFirst few rows of merged data:")
        print(merged_df.head(10))
