import torch
import torch.optim as optim

from modules.neural_network import IntraSectorNN
from modules.loss import calculate_loss


def train_nn(
    features,
    returns,
    epochs=100,
    learning_rate=0.001,
    model=None,
    previous_weights=None,
    temperature=0.5,
    verbose=True
):
    """
    Train the intra-sector neural network.
    """

    if returns.ndim != 2:
        raise ValueError("returns must be a 2-dimensional tensor.")

    num_stocks = returns.shape[1]
    expected_features = num_stocks * 2

    if features.shape[-1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features for {num_stocks} stocks, "
            f"got {features.shape[-1]}."
        )

    if model is None:
        model = IntraSectorNN(
            num_stocks=num_stocks,
            num_features=2,
            temperature=temperature
        )

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    loss_history = []

    for epoch in range(epochs):

        optimizer.zero_grad()

        # NN generates stock weights
        weights = model(features)

        # Convert stock returns into portfolio returns
        portfolio_returns = returns @ weights.squeeze(0)

        # Calculate objective
        loss = calculate_loss(
            portfolio_returns,
            weights.squeeze(0),
            previous_weights=previous_weights
        )

        # Backpropagation
        loss.backward()

        optimizer.step()

        loss_history.append(loss.item())

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"| Loss: {loss.item():.6f}"
            )

    return model, loss_history
