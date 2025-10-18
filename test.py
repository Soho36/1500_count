import pandas as pd

# Use xlsxwriter engine
with pd.ExcelWriter(f"{folder}/{filename}", engine='xlsxwriter') as writer:
    results_df.to_excel(writer, sheet_name="All Runs", index=False)
    summary_df.to_excel(writer, sheet_name="Summary Stats", index=False)
    hist_data.to_excel(writer, sheet_name="Histogram", index=False)

    # === Auto-adjust column widths for all sheets ===

    worksheet = "Summary Stats"
    max_len = max(f[col].astype(str).map(len).max(),len(col)) + 2  # add padding
    worksheet.set_column(i, i, max_len)

