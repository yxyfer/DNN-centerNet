from torch import nn
from ..helpers import Convolution
from _cpools import TopPool, BottomPool, LeftPool, RightPool


class CascadeCornerPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool1: nn.Module, pool2: nn.Module):
        super(CascadeCornerPooling, self).__init__()
        
        self.conv_up = Convolution(in_channels, out_channels)
        self.pool1 = pool1()
        
        self.conv_down = Convolution(in_channels, out_channels)
        
        self.p_conv1 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(out_channels)
        
        self.pool2 = pool2()
        
    def forward(self, x):
        up = self.conv_up(x)
        up = self.pool1(up)
        
        down = self.conv_down(x)
        
        merge = up + down
        merge = self.p_conv1(merge)
        merge = self.p_bn1(merge)
        merge = self.pool2(merge)
        
        return merge

        
class CascadeTLCornerPooling(CascadeCornerPooling):
    def __init__(self, in_channels: int, out_channels: int):
        super(CascadeTLCornerPooling, self).__init__(in_channels, out_channels, LeftPool, TopPool)
        
class CascadeBRCornerPooling(CascadeCornerPooling):
    def __init__(self, in_channels: int, out_channels: int):
        super(CascadeBRCornerPooling, self).__init__(in_channels, out_channels, RightPool, BottomPool)
