"""Leakage-free rolling backtest for the intra-sector neural-network layer."""

import numpy as np
import pandas as pd
import torch

from modules.features import calculate_nn_features
from modules.train_nn import train_nn


def run_rolling_nn_backtest(
    stock_returns: pd.DataFrame,
    training_window: int = 2520,
    feature_lookback: int = 100,
    rebalance_every: int = 20,
    initial_epochs: int = 300,
    update_epochs: int = 25,
    learning_rate: float = 0.001,
    transaction_cost: float = 0.005,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest one sector using only information available at each rebalance.

    A model is first fitted with the previous ``training_window`` observations.
    Its weights are then held for ``rebalance_every`` trading days.  The next
    rebalance retrains the same model using an updated historical window; the
    current day's return is never part of the features or training data.
    """

    stock_returns = stock_returns.dropna().copy()

    if stock_returns.shape[1] != 15:
        raise ValueError(
            "The intra-sector NN requires exactly 15 eligible stocks; "
            f"got {stock_returns.shape[1]}."
        )

    if training_window < feature_lookback:
        raise ValueError("training_window must be at least feature_lookback.")

    if len(stock_returns) <= training_window:
        raise ValueError(
            "Not enough observations for an out-of-sample NN backtest. "
            "Use a longer price-history request or a shorter training window."
        )

    if rebalance_every < 1:
        raise ValueError("rebalance_every must be at least 1.")

    torch.manual_seed(seed)

    model = None
    current_weights = None
    portfolio_returns = []
    benchmark_returns = []
    weight_history = []
    dates = []

    for i in range(training_window, len(stock_returns)):
        is_rebalance = (i - training_window) % rebalance_every == 0
        turnover = 0.0

        if is_rebalance:
            historical_returns = stock_returns.iloc[i - training_window:i]
            features = calculate_nn_features(
                historical_returns.iloc[-feature_lookback:]
            )

            feature_tensor = torch.tensor(
                features,
                dtype=torch.float32
            ).unsqueeze(0)
            return_tensor = torch.tensor(
                historical_returns.values,
                dtype=torch.float32
            )
            previous_tensor = None

            if current_weights is not None:
                previous_tensor = torch.tensor(
                    current_weights,
                    dtype=torch.float32
                )

            epochs = initial_epochs if model is None else update_epochs
            model, _ = train_nn(
                feature_tensor,
                return_tensor,
                epochs=epochs,
                learning_rate=learning_rate,
                model=model,
                previous_weights=previous_tensor,
                verbose=False
            )

            with torch.no_grad():
                new_weights = model(feature_tensor).squeeze(0).numpy()

            if current_weights is not None:
                turnover = float(np.abs(new_weights - current_weights).sum())

            current_weights = new_weights

        daily_returns = stock_returns.iloc[i].to_numpy(dtype=float)
        gross_return = float(np.dot(current_weights, daily_returns))
        net_return = gross_return - transaction_cost * turnover

        portfolio_returns.append(net_return)
        benchmark_returns.append(float(daily_returns.mean()))
        weight_history.append(current_weights.copy())
        dates.append(stock_returns.index[i])

    performance = pd.DataFrame(
        {
            "NN_Return": portfolio_returns,
            "EqualWeight_Return": benchmark_returns,
        },
        index=dates
    )
    performance["NN_Cumulative"] = (1 + performance["NN_Return"]).cumprod() - 1
    performance["EqualWeight_Cumulative"] = (
        (1 + performance["EqualWeight_Return"]).cumprod() - 1
    )

    weights = pd.DataFrame(
        weight_history,
        index=dates,
        columns=stock_returns.columns
    )

    return performance, weights
