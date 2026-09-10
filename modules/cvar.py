import numpy as np


def calculate_var(returns, alpha=0.05):
    """
    Calculate VaR using historical returns.

    VaR is the return threshold below which the worst alpha
    fraction of observations fall.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        raise ValueError("Returns array is empty.")

    return float(np.quantile(returns, alpha))


def calculate_cvar(returns, alpha=0.05):
    """
    Calculate historical CVaR.

    CVaR is the average return of observations that fall
    below or equal to the VaR threshold.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        raise ValueError("Returns array is empty.")

    var = calculate_var(returns, alpha)

    tail_returns = returns[returns <= var]

    if len(tail_returns) == 0:
        return var

    return float(np.mean(tail_returns))


def calculate_partitioned_cvar(returns, alpha=0.05, partitions=5):
    """
    Calculate Partitioned CVaR.

    The return history is divided into contiguous time blocks.
    CVaR is calculated independently for each block and then
    averaged.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        raise ValueError("Returns array is empty.")

    if partitions < 1:
        raise ValueError("Partitions must be at least 1.")

    if len(returns) < partitions:
        raise ValueError(
            "Number of returns must be greater than or equal to partitions."
        )

    blocks = np.array_split(returns, partitions)

    cvar_values = [
        calculate_cvar(block, alpha)
        for block in blocks
    ]

    return float(np.mean(cvar_values))


def calculate_partitioned_portfolio_cvar(
    returns,
    weights,
    alpha=0.05,
    partitions=5
):
    """
    Calculate Partitioned CVaR directly for a portfolio.

    Returns are converted into portfolio returns first,
    then Partitioned CVaR is calculated.
    """

    from modules.portfolio import calculate_portfolio_returns

    portfolio_returns = calculate_portfolio_returns(
        returns,
        weights
    )

    return calculate_partitioned_cvar(
        portfolio_returns,
        alpha,
        partitions
    )