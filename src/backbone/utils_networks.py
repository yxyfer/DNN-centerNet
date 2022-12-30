import torch
from torch import nn

class Convolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(Convolution, self).__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
        
class Residual(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int, kernel_size: int = 3):
        super(Residual, self).__init__()
        
        self.conv1 = Convolution(in_channels, hid_channels, kernel_size)
        
        padding = (kernel_size - 1) // 2
        
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(hid_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.batch_norm(res)
        
        logits = self.relu(x + res)
        
        return logits