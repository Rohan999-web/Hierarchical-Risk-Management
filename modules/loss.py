import torch


def calculate_loss(
    portfolio_returns,
    weights,
    previous_weights=None,
    gamma=3.0,
    lambda_reg=0.01,
    transaction_cost=0.005,
    alpha=0.05,
    partitions=5
):
    """
    Calculate the Stage 3 objective:

    Loss = -Return
           + gamma * Partitioned CVaR
           + lambda * concentration
           + transaction cost * turnover
    """

    # 1. Average portfolio return
    mean_return = torch.mean(portfolio_returns)

    # 2. Partitioned CVaR
    blocks = torch.chunk(portfolio_returns, partitions)

    cvar_values = []

    for block in blocks:
        sorted_returns, _ = torch.sort(block)

        tail_size = max(1, int(len(block) * alpha))

        tail_returns = sorted_returns[:tail_size]

        cvar = torch.mean(tail_returns)

        cvar_values.append(cvar)

    partitioned_cvar = torch.mean(torch.stack(cvar_values))

    # 3. Concentration penalty
    concentration = torch.sum(weights ** 2)

    # 4. Turnover penalty
    if previous_weights is None:
        turnover = torch.tensor(
            0.0,
            device=weights.device
        )
    else:
        turnover = torch.sum(
            torch.abs(weights - previous_weights)
        )

    # 5. Final objective
    loss = (
        -mean_return
        + gamma * (-partitioned_cvar)
        + lambda_reg * concentration
        + transaction_cost * turnover
    )

    return loss