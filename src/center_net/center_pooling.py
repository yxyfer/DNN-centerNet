from torch import nn
from ..helpers import Convolution
from ._cpools import TopPool, BottomPool, LeftPool, RightPool


class CenterPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(CenterPooling, self).__init__()
        
        self.conv_up = Convolution(in_channels, out_channels)
        self.conv_down = Convolution(in_channels, out_channels)
        
        self.top_pool = TopPool()
        self.bottom_pool = BottomPool()
        self.right_pool = RightPool()
        self.left_pool = LeftPool()
        
    def forward(self, x):
        up = self.conv_up(x)
        up = self.left_pool(up)
        up = self.right_pool(up)
        
        down = self.conv_down(x)
        down = self.top_pool(down)
        down = self.bottom_pool(down)
        
        return up + down
        