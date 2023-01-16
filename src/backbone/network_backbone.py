from torch import nn
from ..helpers import Convolution

class BackboneNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            Convolution(1, 32),
            nn.MaxPool2d(2),
            Convolution(32, 64),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 10),
        )
    
    def forward(self, x):
        return self.network(x)