import pandas as pd
import os
import glob

input_dir = "../Optimization_analysis/processed_files"  # folder with xlsx files
output_root = "../Optimization_analysis/month_filtered_results"
os.makedirs(output_root, exist_ok=True)

# Define the month columns in order for sorting
month_columns = [
    'TradeJanuary',
    'TradeFebruary',
    'TradeMarch',
    'TradeApril',
    'TradeMay',
    'TradeJune',
    'TradeJuly',
    'TradeAugust',
    'TradeSeptember',
    'TradeOctober',
    'TradeNovember',
    'TradeDecember'
]

# Create a mapping for month sorting
month_order = {month: i for i, month in enumerate(month_columns)}

# Initialize a list to collect all dataframes for merging
all_merged_data = []

# Process file by file
for input_file_path in glob.glob(os.path.join(input_dir, "*.xlsx")):
    input_file_name = os.path.basename(input_file_path)

    print(f"\nProcessing {input_file_name}")

    # Read the Excel file
    df = pd.read_excel(input_file_path)

    # Ensure month columns are boolean
    for col in month_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.lower().map({'true': True, 'false': False})
            df[col] = df[col].astype(bool)

    # For each month, find rows where only that month is true
    for month in month_columns:
        mask = df[month] == True
        other_months = [m for m in month_columns if m != month]
        for other_month in other_months:
            mask = mask & (df[other_month] == False)

        matching_rows = df[mask]

        if not matching_rows.empty:
            print(f"Found {len(matching_rows)} rows for {month}.")
        else:
            print(f"No rows found for {month}")

    # Combined summary
    all_matches = pd.DataFrame()

    for month in month_columns:
        mask = df[month] == True
        other_months = [m for m in month_columns if m != month]
        for other_month in other_months:
            mask = mask & (df[other_month] == False)

        matching_rows = df[mask].copy()
        if not matching_rows.empty:
            matching_rows['TargetMonth'] = month
            all_matches = pd.concat([all_matches, matching_rows], ignore_index=True)

    if not all_matches.empty:
        # Save individual file results
        output_filename = f"{input_file_name}_all_month_specific_rows.xlsx"
        output_path = os.path.join(output_root, output_filename)
        all_matches.to_excel(output_path, index=False)
        print(f"Combined summary saved to {output_path}")

        # Add source file column and collect for merging
        all_matches['SourceFile'] = output_filename
        all_merged_data.append(all_matches)

# Create merged file if any data was collected
if all_merged_data:
    merged_df = pd.concat(all_merged_data, ignore_index=True)

    # Add a month order column for sorting
    merged_df['MonthOrder'] = merged_df['TargetMonth'].map(month_order)

    # Sort by month order, then by SourceFile
    merged_df = merged_df.sort_values(['MonthOrder', 'SourceFile'])

    # Remove the temporary MonthOrder column
    merged_df = merged_df.drop('MonthOrder', axis=1)

    # Reorder columns to have SourceFile as the first column
    cols = ['SourceFile'] + [col for col in merged_df.columns if col != 'SourceFile']
    merged_df = merged_df[cols]

    # Save merged file
    merged_output_path = os.path.join(output_root, "ALL_FILES_MERGED_month_specific_rows.xlsx")
    merged_df.to_excel(merged_output_path, index=False)
    print(f"\nMerged file saved to: {merged_output_path}")
    print(f"Total rows in merged file: {len(merged_df)}")

    # Display summary of rows per month
    print("\nRows per month in merged file:")
    month_counts = merged_df['TargetMonth'].value_counts().sort_index()
    for month, count in month_counts.items():
        print(f"{month}: {count} rows")
else:
    print("\nNo data found to merge.")

print("\nProcessing complete!")