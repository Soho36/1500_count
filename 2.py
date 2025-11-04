start_idx = 0
while start_idx < len(dataframe):
    cumulative_pnl = 0
    min_cumulative_pnl = 0
    days = 0
    reached = False
    blown = False

    # --- reset trailing DD vars per run ---
    peak_pnl = 0
    trailing_floor = -MAX_DD

    # --- dynamic lot setup ---
    contracts = SIZE if not USE_DYNAMIC_LOT else 1
    contract_history = []

    for i in range(start_idx, len(dataframe)):
        # Record contract size
        contract_history.append(contracts)

        # --- Apply today's PnL ---
        cumulative_pnl_today = dataframe.loc[i, 'P/L (Net)'] * contracts
        projected_pnl = cumulative_pnl + cumulative_pnl_today
        days += 1

        # --- Check if today overshoots the target ---
        if projected_pnl >= TARGET:
            # take only the remaining amount needed to reach target
            cumulative_pnl_today = TARGET - cumulative_pnl
            cumulative_pnl = TARGET
            min_cumulative_pnl = min(min_cumulative_pnl, cumulative_pnl)

            # log the truncated last step
            if SAVE_CONTRACT_LOG and start_idx < MAX_RUNS_TO_LOG:
                detailed_log_helper(detailed_log, dataframe, start_idx, i, contracts,
                                    cumulative_pnl_today, cumulative_pnl, peak_pnl, trailing_floor)

            # record the completed run
            results.append({
                "Start_Date": dataframe.loc[start_idx, 'Date'],
                "Rows_to_+Target": days,
                "Rows_to_blown": None,
                "Max_Drawdown": abs(min_cumulative_pnl),
                "Average_Contracts": sum(contract_history) / len(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "Minimum_Contracts": min(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "Maximum_Contracts": max(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "End_Date": dataframe.loc[i, 'Date'],
                "Blown": False
            })
            reached = True
            break

        # --- otherwise continue normally ---
        cumulative_pnl = projected_pnl
        min_cumulative_pnl = min(min_cumulative_pnl, cumulative_pnl)

        # --- Update contract size dynamically (only if enabled) ---
        if USE_DYNAMIC_LOT:
            contracts = max(1, 1 + int(cumulative_pnl // CONTRACT_STEP))

        # --- Update DD logic (using intraday highs if available) ---
        if USE_TRAILING_DD:
            if "Hi" in dataframe.columns and "Net" in dataframe.columns:
                intraday_peak = cumulative_pnl + (dataframe.loc[i, "Hi"] - dataframe.loc[i, "Close"])
                peak_pnl = max(peak_pnl, intraday_peak)
            else:
                peak_pnl = max(peak_pnl, cumulative_pnl)

            trailing_floor = peak_pnl - MAX_DD
            dd_limit = trailing_floor
            dd_breached = cumulative_pnl < dd_limit
        else:
            dd_limit = -MAX_DD
            dd_breached = cumulative_pnl <= dd_limit

        # --- save per-day details ---
        if SAVE_CONTRACT_LOG and start_idx < MAX_RUNS_TO_LOG:
            detailed_log_helper(detailed_log, dataframe, start_idx, i, contracts,
                                cumulative_pnl_today, cumulative_pnl, peak_pnl, trailing_floor)

        # --- Check blowup condition ---
        if dd_breached:
            # Cut today’s loss to land exactly on the drawdown limit
            adjustment = dd_limit - cumulative_pnl
            cumulative_pnl_today += adjustment
            cumulative_pnl = dd_limit  # precisely hit DD threshold

            # Log the truncated last step
            if SAVE_CONTRACT_LOG and start_idx < MAX_RUNS_TO_LOG:
                detailed_log_helper(detailed_log, dataframe, start_idx, i, contracts,
                                    cumulative_pnl_today, cumulative_pnl, peak_pnl, trailing_floor)

            results.append({
                "Start_Date": dataframe.loc[start_idx, 'Date'],
                "Rows_to_+Target": None,
                "Rows_to_blown": days,
                "Max_Drawdown": peak_pnl - cumulative_pnl if USE_TRAILING_DD else abs(min_cumulative_pnl),
                "Average_Contracts": sum(contract_history) / len(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "Minimum_Contracts": min(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "Maximum_Contracts": max(contract_history) if USE_DYNAMIC_LOT else SIZE,
                "End_Date": dataframe.loc[i, 'Date'],
                "Blown": True
            })
            blown = True
            break

    # --- If we reach the end without hitting either condition ---
    if not reached and not blown:
        results.append({
            "Start_Date": dataframe.loc[start_idx, 'Date'],
            "Rows_to_+Target": None,
            "Rows_to_blown": None,
            "Max_Drawdown": abs(min_cumulative_pnl),
            "Average_Contracts": sum(contract_history) / len(contract_history) if USE_DYNAMIC_LOT else SIZE,
            "Minimum_Contracts": min(contract_history) if USE_DYNAMIC_LOT else SIZE,
            "Maximum_Contracts": max(contract_history) if USE_DYNAMIC_LOT else SIZE,
            "End_Date": None,
            "Blown": False
        })

    # --- Advance start index depending on mode ---
    if OVERLAPPING_RUNS:
        start_idx += 1  # traditional mode — start next day
    else:
        # jump to the next day after this run ended
        if reached or blown:
            start_idx = i + 1
        else:
            start_idx += 1  # if it never finished, just move by one anyway
