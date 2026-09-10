import numpy as np
import pandas as pd


def calculate_portfolio_returns(returns, weights):
    """
    Calculate portfolio returns from asset returns and weights.

    Parameters
    ----------
    returns : pd.DataFrame or np.ndarray
        Rows = dates
        Columns = stocks/assets

    weights : array-like
        Portfolio weight for each stock.

    Returns
    -------
    pd.Series or np.ndarray
        Portfolio return for each date.
    """

    if isinstance(returns, pd.DataFrame):
        returns_array = returns.values
        index = returns.index
    else:
        returns_array = np.asarray(returns, dtype=float)
        index = None

    weights = np.asarray(weights, dtype=float)

    if returns_array.ndim != 2:
        raise ValueError("Returns must be a 2-dimensional matrix.")

    if len(weights) != returns_array.shape[1]:
        raise ValueError(
            "Number of weights must match number of assets."
        )

    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    portfolio_returns = returns_array @ weights

    if index is not None:
        return pd.Series(
            portfolio_returns,
            index=index,
            name="Portfolio_Return"
        )

    return portfolio_returns