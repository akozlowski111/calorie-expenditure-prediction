import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
