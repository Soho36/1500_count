import pandas as pd
import os
import glob


def process_xls_files_with_backup(input_files_path, output_files_path):
    os.makedirs(output_files_path, exist_ok=True)

    xls_files = glob.glob(os.path.join(input_files_path, "*.xls"))

    if not xls_files:
        print(f"No .xls files found in {input_files_path}")
        return

    print(f"Found {len(xls_files)} .xls file(s) to process")

    for file in xls_files:
        try:
            df = pd.read_excel(file, engine="xlrd")
            print(f"Processing: {os.path.basename(file)}")

            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].replace({
                        "true": True,
                        "false": False,
                        "TRUE": True,
                        "FALSE": False
                    })

            output_file = os.path.join(
                output_files_path,
                os.path.basename(file).replace(".xls", ".xlsx")
            )

            df.to_excel(output_file, index=False)
            print(f"  ✓ Saved: {os.path.basename(output_file)}")

        except Exception as e:
            print(f"  ✗ Error processing {file}: {e}")
            return


if __name__ == "__main__":
    # Example usage:

    input_files_folder = "Optimization_analysis/input_files_time_windows"  # Current directory
    output_files_folder = "../Optimization_analysis/processed_files/"  # Output directory
    # Method 1: Process all files in current directory
    # process_xls_files(current_folder)

    # Method 2: Process with backup
    process_xls_files_with_backup(input_files_folder, output_files_folder)
