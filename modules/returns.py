import numpy as np
import pandas as pd


def compute_stock_returns(prices: pd.DataFrame):
    """
    Convert stock prices into daily log returns and simple returns.

    Log returns:
        Used for DCC-GARCH modelling.

    Simple returns:
        Used for portfolio P&L / backtesting.
    """

    log_returns = np.log(prices / prices.shift(1)).dropna()

    simple_returns = (
        prices / prices.shift(1) - 1
    ).dropna()

    return log_returns, simple_returns


def compute_sector_returns(
    prices: pd.DataFrame,
    mapping: dict
):
    """
    Calculate equal-weighted returns for each sector.

    Only stocks that are actually present in the cleaned
    price matrix are included.
    """

    log_stock_returns, simple_stock_returns = (
        compute_stock_returns(prices)
    )

    sector_log = {}
    sector_simple = {}

    for sector, tickers in mapping.items():

        # Keep only stocks with available pricing data
        valid_tickers = [
            ticker
            for ticker in tickers
            if ticker in prices.columns
        ]

        if not valid_tickers:
            print(
                f"WARNING: No valid stocks found "
                f"for sector '{sector}'."
            )
            continue

        # Equal weight every available stock
        weight = 1.0 / len(valid_tickers)

        sector_log[sector] = (
            log_stock_returns[valid_tickers]
            .sum(axis=1) * weight
        )

        sector_simple[sector] = (
            simple_stock_returns[valid_tickers]
            .sum(axis=1) * weight
        )

    sector_log_returns = pd.DataFrame(sector_log).dropna()
    sector_simple_returns = pd.DataFrame(sector_simple).dropna()

    return sector_log_returns, sector_simple_returns