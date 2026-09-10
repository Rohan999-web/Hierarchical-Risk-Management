import torch
import torch.nn as nn


class IntraSectorNN(nn.Module):
    def __init__(self, num_stocks=15, num_features=2):
        super().__init__()

        input_size = num_stocks * num_features

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.SiLU(),

            nn.Linear(64, 64),
            nn.SiLU(),

            nn.Linear(64, num_stocks)
        )

    def forward(self, x):
        logits = self.network(x)

        # Convert outputs into portfolio weights
        weights = torch.softmax(logits, dim=-1)

        return weights