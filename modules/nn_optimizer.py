import torch

from modules.features import calculate_nn_features
from modules.neural_network import IntraSectorNN


def generate_sector_weights(
    stock_returns,
    lookback=100,
    temperature=0.5
):
    """
    Generate NN-based stock weights for one sector.

    Parameters
    ----------
    stock_returns : pd.DataFrame
        Historical returns for stocks in one sector.

    lookback : int
        Number of trading days used for NN features.

    temperature : float
        Softmax temperature.

    Returns
    -------
    pd.Series
        NN-generated weight for each stock.
    """

    num_stocks = stock_returns.shape[1]

    if num_stocks != 15:
        raise ValueError(
            f"Expected 15 stocks for the current pipeline, "
            f"got {num_stocks}."
        )

    # Create 100-day × 2 feature vector
    features = calculate_nn_features(
        stock_returns,
        lookback
    )

    # Convert features to PyTorch tensor
    features_tensor = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0)

    # Create NN
    model = IntraSectorNN(
        num_stocks=num_stocks,
        num_features=2,
        temperature=temperature
    )

    # Generate weights
    with torch.no_grad():
        weights = model(features_tensor).squeeze(0)

    return weights.numpy()
