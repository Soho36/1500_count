import sys
import pandas as pd
from pathlib import Path
import numpy as np

# ==============================================================================
#  CONFIG — edit these values before running
# ==============================================================================

INPUT_FILE  = r"C:\Source\DATABENTO\MNQ\seconds\MNQ_ohlcv-1s.csv"        # Path to your 1s OHLC source CSV
OUTPUT_FILE = r"C:\Source\DATABENTO\MNQ\seconds\MT5_MNQ_ohlcv-1s_ticks_silla.csv"     # Path for the output MT5 tick CSV
DOMINANT_FILE = r"C:\Source\DATABENTO\MNQ\seconds\dominant_contracts.csv"

# --- Column names in your source CSV ---
# If date and time are in SEPARATE columns:
DATE_COL    = None
TIME_COL    = None
OPEN_COL  = "open"
HIGH_COL  = "high"
LOW_COL   = "low"
CLOSE_COL = "close"
SYMBOL_COL = "symbol"
VOLUME_COL = "volume"


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
    df_rep["step"] = np.tile(np.arange(4), len(df))

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

    # step must be 0..3
    # df_rep["step"] = np.tile(np.arange(4), len(df))

    # millisecond offsets
    ms_offsets = np.tile([0, 250, 500, 750], len(df))

    dt_with_ms = df_rep["_dt"] + pd.to_timedelta(ms_offsets, unit="ms")

    df_rep["<TIME>"] = dt_with_ms.dt.strftime("%H:%M:%S.%f").str[:-3]

    # spread (centered)
    df_rep["<BID>"] = df_rep["price"] - SPREAD
    df_rep["<ASK>"] = df_rep["price"] + SPREAD
    df_rep["<LAST>"] = df_rep["price"]
    df_rep["<VOLUME>"] = VOLUME

    return df_rep[[
        "<DATE>", "<TIME>", "<BID>", "<ASK>", "<LAST>", "<VOLUME>"
    ]]


def load_dominant_map():
    df = pd.read_csv(DOMINANT_FILE)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return dict(zip(df["trade_date"], df["symbol"]))


def process_in_chunks():
    import time

    start_time = time.perf_counter()
    dominant_dict = load_dominant_map()

    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    chunk_size = 200_000

    reader = pd.read_csv(
        input_path,
        sep=CSV_SEP,
        chunksize=chunk_size,
        usecols=[DATETIME_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, SYMBOL_COL]
    )

    total_ticks = 0
    first_chunk = True
    last_dt = None

    # FIX: accumulate a small overlap buffer across chunks
    carry_df = pd.DataFrame()

    for i, df_chunk in enumerate(reader):
        print(f"[+] Processing chunk {i+1}...")

        df_chunk["_dt"] = pd.to_datetime(
            df_chunk[DATETIME_COL],
            format=DATETIME_FMT,
            utc=True
        )
        df_chunk["_dt"] = df_chunk["_dt"].dt.tz_convert("America/Chicago")
        df_chunk["_dt"] = df_chunk["_dt"].dt.tz_localize(None)
        df_chunk["_dt"] = df_chunk["_dt"] + pd.Timedelta(hours=8)

        df_chunk["trade_date"] = df_chunk["_dt"].dt.date

        # FIX: map dominant per-row using the shifted trade_date (same as dominant builder)
        df_chunk["_dominant"] = df_chunk["trade_date"].map(dominant_dict)

        missing = df_chunk["_dominant"].isna().sum()
        if missing > 0:
            print(f"[!] Warning: {missing} rows with no dominant contract in chunk {i+1} — dropping")

        # FIX: keep rows where symbol matches the dominant for THAT row's trade_date
        df_chunk = df_chunk[df_chunk[SYMBOL_COL] == df_chunk["_dominant"]].copy()

        df_chunk = df_chunk[["_dt", OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]].copy()
        df_chunk.columns = ["_dt", "open", "high", "low", "close"]

        df_chunk[["open", "high", "low", "close"]] = df_chunk[
            ["open", "high", "low", "close"]
        ].astype(float)

        # FIX: prepend carry from previous chunk before deduplication
        if not carry_df.empty:
            df_chunk = pd.concat([carry_df, df_chunk], ignore_index=True)

        df_chunk = df_chunk.sort_values("_dt") \
                           .drop_duplicates(subset="_dt", keep="last") \
                           .reset_index(drop=True)

        # FIX: strict cut — never re-process anything at or before last written timestamp
        if last_dt is not None:
            df_chunk = df_chunk[df_chunk["_dt"] > last_dt].reset_index(drop=True)

        if df_chunk.empty:
            carry_df = pd.DataFrame()
            continue

        # FIX: save the last 4 rows as carry for next chunk (one full "second" worth of ticks)
        carry_df = df_chunk.tail(4).copy()
        df_chunk = df_chunk.iloc[:-4].reset_index(drop=True)

        if df_chunk.empty:
            continue

        ticks = build_ticks_fast(df_chunk)

        # FIX: sort by actual datetime, not strings
        ticks["_sort_dt"] = pd.to_datetime(
            ticks["<DATE>"] + " " + ticks["<TIME>"],
            format="%Y.%m.%d %H:%M:%S.%f"
        )
        ticks = ticks.sort_values("_sort_dt").drop(columns=["_sort_dt"]).reset_index(drop=True)

        ticks.to_csv(
            output_path,
            mode="a",
            header=first_chunk,
            index=False
        )
        first_chunk = False
        total_ticks += len(ticks)
        last_dt = df_chunk["_dt"].iloc[-1]
        print(f"    ticks written: {total_ticks:,}")

    # FIX: flush the carry buffer (last few rows never got written)
    if not carry_df.empty:
        if last_dt is not None:
            carry_df = carry_df[carry_df["_dt"] > last_dt].reset_index(drop=True)
        if not carry_df.empty:
            ticks = build_ticks_fast(carry_df)
            ticks.to_csv(output_path, mode="a", header=False, index=False)
            total_ticks += len(ticks)
            print(f"    [carry flush] ticks written: {total_ticks:,}")

    end_time = time.perf_counter()
    print(f"[+] Done. Total ticks: {total_ticks:,}")
    print(f"⏱ Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    process_in_chunks()
