from torch import nn
from ..helpers import Convolution

def _conv_layer(in_channels: int, hidden_channels: int, out_channels: int):
    return nn.Sequential(
        Convolution(in_channels, hidden_channels, with_bn=False),
        nn.Conv2d(hidden_channels, out_channels, (1, 1))   
    )

class HeatMapLayer(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        
        self.heatmap = _conv_layer(in_channels, hidden_channels, out_channels)
        
    def forward(self, x):
        return self.heatmap(x)
    
class EmbeddingLayer(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 1):
        super().__init__()
        
        self.embedding = _conv_layer(in_channels, hidden_channels, out_channels)
        
    def forward(self, x):
        return self.embedding(x)

class OffsetLayer(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        
        self.offset = _conv_layer(in_channels, hidden_channels, out_channels)
        
    def forward(self, x):
        return self.offset(x)