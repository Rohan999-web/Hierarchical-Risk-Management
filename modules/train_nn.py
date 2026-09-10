import torch
import torch.optim as optim

from modules.neural_network import IntraSectorNN
from modules.loss import calculate_loss


def train_nn(
    features,
    returns,
    epochs=100,
    learning_rate=0.001
):
    """
    Train the intra-sector neural network.
    """

    model = IntraSectorNN(
        num_stocks=15,
        num_features=2
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
            weights.squeeze(0)
        )

        # Backpropagation
        loss.backward()

        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"| Loss: {loss.item():.6f}"
            )

    return model, loss_history