# Initial state
peak_pnl = 1000
intraday_peak = 1200

peak_pnl = max(peak_pnl, intraday_peak)
print(peak_pnl)

# Later, if we get a lower peak
intraday_peak = 1100
peak_pnl = max(peak_pnl, intraday_peak)
print(peak_pnl)

# If we get a new higher peak
intraday_peak = 1500
peak_pnl = max(peak_pnl, intraday_peak)
print(peak_pnl)
