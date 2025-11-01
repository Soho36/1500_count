# --- Update DD logic (using intraday highs if available) ---
if USE_TRAILING_DD:
    if "Hi" in df.columns and "Net" in df.columns:
        intraday_peak = cumulative_pnl + (df.loc[i, "Hi"] - df.loc[i, "Close"])
        peak_pnl = max(peak_pnl, intraday_peak)
    else:
        peak_pnl = max(peak_pnl, cumulative_pnl)

    trailing_floor = peak_pnl - MAX_DD
    dd_breached = cumulative_pnl < trailing_floor
else:
    dd_breached = cumulative_pnl <= -MAX_DD

# --- Check blowup condition ---
if dd_breached:
    results.append({
        "Start_Date": df.loc[start_idx, 'Date'],
        "Rows_to_+Target": None,
        "Rows_to_blown": days,
        "Max_Drawdown": peak_pnl - cumulative_pnl if USE_TRAILING_DD else abs(min_cumulative_pnl),
        "Average_Contracts": sum(contract_history) / len(contract_history) if USE_DYNAMIC_LOT else SIZE,
        "Minimum_Contracts": min(contract_history) if USE_DYNAMIC_LOT else SIZE,
        "Maximum_Contracts": max(contract_history) if USE_DYNAMIC_LOT else SIZE,
        "End_Date": df.loc[i, 'Date'],
        "Blown": True
    })
    blown = True
    break