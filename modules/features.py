import numpy as np
import pandas as pd


def calculate_nn_features(returns, lookback=100):
    """
    Calculate the two features used by the intra-sector NN:

    1. Average return over the lookback period
    2. Volatility over the lookback period

    Parameters
    ----------
    returns : pd.DataFrame
        Rows = dates
        Columns = stocks

    lookback : int
        Number of historical trading days.

    Returns
    -------
    np.ndarray
        Flattened feature vector:
        [mean_return_stock_1, volatility_stock_1,
        mean_return_stock_2, volatility_stock_2, ...]
    """

    if len(returns) < lookback:
        raise ValueError(
            f"Need at least {lookback} observations, "
            f"got {len(returns)}."
        )

    window = returns.iloc[-lookback:].copy()

    mean_returns = window.mean()
    volatility = window.std()

    features = []

    for stock in returns.columns:
        features.append(mean_returns[stock])
        features.append(volatility[stock])

    features = np.asarray(features, dtype=np.float32)

    # Scale NN input features as specified in the report
    features = features * 100.0

    return features
