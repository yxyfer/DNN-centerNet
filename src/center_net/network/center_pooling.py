from torch import nn
from ...helpers import Convolution
from .._cpools import TopPool, BottomPool, LeftPool, RightPool


class CenterPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(CenterPooling, self).__init__()
        
        self.conv_up = Convolution(in_channels, out_channels)
        self.conv_down = Convolution(in_channels, out_channels)
        
        self.top_pool = TopPool()
        self.bottom_pool = BottomPool()
        self.right_pool = RightPool()
        self.left_pool = LeftPool()
        
        self.p_conv1 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(out_channels)
        
        self.conv1 = nn.Conv2d(out_channels, out_channels, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = Convolution(out_channels, out_channels)
        
    def forward(self, x):
        up = self.conv_up(x)
        up = self.left_pool(up)
        up = self.right_pool(up)
        
        down = self.conv_down(x)
        down = self.top_pool(down)
        down = self.bottom_pool(down)
        
        merge = self.p_conv1(up + down)
        merge = self.p_bn1(merge)
        
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(merge + bn1)

        conv2 = self.conv2(relu1)
        return conv2
        