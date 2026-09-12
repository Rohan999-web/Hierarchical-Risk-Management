import torch
import torch.nn as nn


class IntraSectorNN(nn.Module):

    def __init__(
        self,
        num_stocks=15,
        num_features=2,
        temperature=0.5
    ):
        super().__init__()

        input_size = num_stocks * num_features

        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, num_stocks)

        self.activation = nn.SiLU()

        self.temperature = temperature
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.activation(
            self.layer1(x)
        )

        x = self.activation(
            self.layer2(x)
        )

        logits = self.layer3(x)

        weights = self.softmax(
            logits / self.temperature
        )

        return weights