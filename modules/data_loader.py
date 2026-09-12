import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


def load_universe_mapping(filepath: str) -> dict:
    """
    Read the stock universe file and group tickers by sector.

    Expected columns:
        Ticker
        Sector
    """

    print(f"Reading stock universe from '{filepath}'...")

    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)

    elif filepath.endswith((".xls", ".xlsx")):
        df = pd.read_excel(filepath)

    else:
        raise ValueError(
            "Unsupported file format. Please provide a .csv or .xlsx file."
        )

    # Clean column names
    df.columns = df.columns.str.strip().str.title()

    if "Ticker" not in df.columns or "Sector" not in df.columns:
        raise ValueError(
            "File must contain 'Ticker' and 'Sector' columns."
        )

    universe = {}

    for sector, group in df.groupby("Sector"):
        tickers = group["Ticker"].astype(str).str.strip().tolist()
        universe[sector] = tickers

    total_stocks = sum(len(tickers) for tickers in universe.values())

    print(
        f"Successfully loaded {total_stocks} stocks "
        f"across {len(universe)} sectors."
    )

    return universe


def download_yahoo_data(
    tickers: list,
    years: int = 5,
    cache_dir: str = "yf_cache",
    refresh_short_cache: bool = True
) -> pd.DataFrame:
    """
    Download historical daily closing prices from Yahoo Finance.

    Successfully downloaded stocks are cached locally so that
    future runs don't need to download them again.
    """

    print(f"\nInitiating data fetch for {len(tickers)} stocks...")
    print(f"Caching files in './{cache_dir}'")

    os.makedirs(cache_dir, exist_ok=True)

    end_date = datetime.today()
    start_date = end_date - timedelta(
        days=int(years * 365.25) + 30
    )

    df_list = []
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):

        safe_filename = ticker.replace("&", "_amp_") + ".csv"
        cache_path = os.path.join(cache_dir, safe_filename)

        # -------------------------------------------------
        # 1. Try cached data first
        # -------------------------------------------------
        if os.path.exists(cache_path):

            try:
                series = pd.read_csv(
                    cache_path,
                    index_col=0,
                    parse_dates=True
                ).iloc[:, 0]

                series = series.dropna()

                # Check whether cached data covers the requested period
                cached_start = series.index.min()

                # Yahoo only returns trading days.  A request starting on a
                # weekend or market holiday can legitimately begin a few
                # calendar days later, so allow that small market-calendar
                # gap before deciding the cache is incomplete.
                required_start = pd.Timestamp(start_date)
                first_acceptable_date = required_start + pd.Timedelta(days=7)

                if cached_start <= first_acceptable_date:
                    series.name = ticker
                    df_list.append(series)
                    continue

                if not refresh_short_cache:
                    series.name = ticker
                    df_list.append(series)
                    continue

                print(
                    f"  Cache for {ticker} is too short. "
                    "Re-downloading..."
                )

            except Exception:
                print(
                    f"  Corrupted cache for {ticker}. "
                    "Re-downloading..."
                )

        # -------------------------------------------------
        # 2. Download from Yahoo Finance
        # -------------------------------------------------
        try:

            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )

            if data.empty:
                failed_tickers.append(ticker)
                continue

            close_series = data["Close"].copy()

            # yfinance can sometimes return a DataFrame
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]

            close_series.name = ticker

            # Remove missing values
            close_series = close_series.dropna()

            # Save cache
            close_series.to_csv(
                cache_path,
                header=[ticker]
            )

            df_list.append(close_series)

            # Small delay between requests
            time.sleep(np.random.uniform(0.3, 0.7))

        except Exception as e:

            failed_tickers.append(ticker)
            print(f"\n  Failed: {ticker} → {e}")

        # Progress
        sys.stdout.write(
            f"\r  Processed {i}/{len(tickers)} assets..."
        )
        sys.stdout.flush()

    print("\n\nConsolidating stock data...")

    if not df_list:
        raise RuntimeError(
            "No stock data could be downloaded."
        )

    if failed_tickers:
        print(
            f"Skipped {len(failed_tickers)} stocks: "
            f"{failed_tickers[:8]}"
        )

    # -------------------------------------------------
    # 3. Combine all stocks
    # -------------------------------------------------

    wide_df = pd.concat(
        df_list,
        axis=1
    ).sort_index()

    # -------------------------------------------------
    # 4. Remove stocks with too much missing data
    # -------------------------------------------------

    threshold = int(len(wide_df) * 0.90)

    valid_cols = wide_df.dropna(
        thresh=threshold,
        axis=1
    ).columns

    dropped = set(wide_df.columns) - set(valid_cols)

    if dropped:
        print(
            f"Cleaned out {len(dropped)} stocks "
            "due to incomplete history."
        )

    wide_df = wide_df[valid_cols]

    # Fill small gaps
    wide_df = wide_df.ffill().bfill()

    print(
        f"Final Active Universe: "
        f"{wide_df.shape[1]} stocks × "
        f"{wide_df.shape[0]} trading days."
    )

    return wide_df
