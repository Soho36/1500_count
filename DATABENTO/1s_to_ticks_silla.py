import sys
import pandas as pd
from pathlib import Path
import numpy as np

# ==============================================================================
#  CONFIG — edit these values before running
# ==============================================================================

INPUT_FILE  = r"C:\Source\DATABENTO\MNQ\seconds\MNQ_ohlcv-1s.csv"        # Path to your 1s OHLC source CSV
OUTPUT_FILE = r"C:\Source\DATABENTO\MNQ\seconds\MT5_MNQ_ohlcv-1s_ticks2.csv"     # Path for the output MT5 tick CSV

# --- Column names in your source CSV ---
# If date and time are in SEPARATE columns:
DATE_COL    = None
TIME_COL    = None
OPEN_COL  = "open"
HIGH_COL  = "high"
LOW_COL   = "low"
CLOSE_COL = "close"

# --- Tick construction ---
SPREAD    = 0.25      # MNQ realistic approximation (can set 0.0)
"""
Sixth, strategy-level bias (important reminder)
Current logic:
bullish → [o, l, h, c]
bearish → [o, h, l, c]
This introduces structural bias:
bullish candles → SL-first tendency
bearish candles → TP-first tendency
For a breakout system, this can materially affect results.
At minimum, should later test:
PATH_MODE = "ohlc"
PATH_MODE = "olhc"
And compare outcomes.
Better solution (later): randomized ordering.
"""
PATH_MODE = "auto"    # "auto", "ohlc", "olhc"

# If date and time are in a SINGLE combined column, set this instead:
DATETIME_COL = "ts_event"         # e.g. "Datetime"  — set to None to use DATE_COL + TIME_COL
DATETIME_FMT = None               # None = let pandas auto-detect the format (handles ISO 8601 + nanoseconds)

# --- Other settings ---
CSV_SEP     = ","                 # CSV delimiter (use "\t" for tab-separated)
VOLUME      = 1                   # Volume value written to every tick row

# ==============================================================================


MT5_DATE_FMT = "%Y.%m.%d"
MT5_TIME_FMT = "%H:%M:%S"


def load_ohlc() -> pd.DataFrame:
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        sys.exit(f"[!] Input file not found: {INPUT_FILE}")

    print(f"[+] Reading: {INPUT_FILE}")
    df = pd.read_csv(input_path, sep=CSV_SEP, skipinitialspace=True)

    if DATETIME_COL:
        df["_dt"] = pd.to_datetime(df[DATETIME_COL], format=DATETIME_FMT, utc=True).dt.tz_localize(None)
    else:
        df["_dt"] = pd.to_datetime(df[DATE_COL].astype(str) + " " + df[TIME_COL].astype(str))

    required = [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]
    for col in required:
        if col not in df.columns:
            sys.exit(f"[!] Column '{col}' not found in CSV.")

    df = df[["_dt", OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]].copy()
    df.columns = ["_dt", "open", "high", "low", "close"]

    df = df.sort_values("_dt").reset_index(drop=True)
    return df


def build_ticks_fast(df: pd.DataFrame) -> pd.DataFrame:
    # repeat each row 4 times (for O,H,L,C path)
    df_rep = df.loc[df.index.repeat(4)].copy()

    # position inside each second (0,1,2,3)
    df_rep["step"] = df_rep.groupby(level=0).cumcount()

    # build path
    o = df_rep["open"].values
    h = df_rep["high"].values
    l = df_rep["low"].values
    c = df_rep["close"].values
    step = df_rep["step"].values

    if PATH_MODE == "auto":
        bullish_base = (df["close"].values >= df["open"].values)
        bullish = np.repeat(bullish_base, 4)

        price = np.where(
            step == 0, o,
            np.where(
                step == 1,
                np.where(bullish, l, h),
                np.where(
                    step == 2,
                    np.where(bullish, h, l),
                    c
                )
            )
        )
    elif PATH_MODE == "ohlc":
        price = np.where(
            step == 0, o,
            np.where(step == 1, h,
            np.where(step == 2, l, c))
        )
    elif PATH_MODE == "olhc":
        price = np.where(
            step == 0, o,
            np.where(step == 1, l,
            np.where(step == 2, h, c))
        )
    else:
        raise ValueError("Invalid PATH_MODE")

    df_rep["price"] = price

    # time (no ms needed for MT5)
    df_rep["<DATE>"] = df_rep["_dt"].dt.strftime(MT5_DATE_FMT)
    df_rep["<TIME>"] = df_rep["_dt"].dt.strftime(MT5_TIME_FMT)

    # spread (centered)
    df_rep["<BID>"] = df_rep["price"] - SPREAD
    df_rep["<ASK>"] = df_rep["price"] + SPREAD
    df_rep["<LAST>"] = df_rep["price"]
    df_rep["<VOLUME>"] = VOLUME

    return df_rep[[
        "<DATE>", "<TIME>", "<BID>", "<ASK>", "<LAST>", "<VOLUME>"
    ]]


def process_in_chunks():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # remove file if exists
    if output_path.exists():
        output_path.unlink()

    chunk_size = 200_000  # adjust if needed

    reader = pd.read_csv(
        input_path,
        sep=CSV_SEP,
        chunksize=chunk_size
    )

    total_ticks = 0

    first_chunk = True
    for i, df_chunk in enumerate(reader):
        print(f"[+] Processing chunk {i+1}...")

        # --- datetime ---
        if DATETIME_COL:
            df_chunk["_dt"] = pd.to_datetime(
                df_chunk[DATETIME_COL],
                format=DATETIME_FMT,
                utc=True
            ).dt.tz_localize(None)
        else:
            df_chunk["_dt"] = pd.to_datetime(
                df_chunk[DATE_COL].astype(str) + " " + df_chunk[TIME_COL].astype(str)
            )

        df_chunk = df_chunk[["_dt", OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]].copy()
        df_chunk.columns = ["_dt", "open", "high", "low", "close"]

        df_chunk[["open", "high", "low", "close"]] = df_chunk[
            ["open", "high", "low", "close"]
        ].astype(float)

        ticks = build_ticks_fast(df_chunk)

        # append to file
        ticks.to_csv(
            output_path,
            mode="a",
            header=first_chunk,
            index=False
        )
        first_chunk = False

        total_ticks += len(ticks)
        print(f"    ticks written: {total_ticks:,}")

    print(f"[+] Done. Total ticks: {total_ticks:,}")


if __name__ == "__main__":
    process_in_chunks()
