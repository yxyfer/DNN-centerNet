from torch import nn
from ...helpers import Convolution
from .._cpools import TopPool, BottomPool, LeftPool, RightPool


class CascadeCornerPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool1: nn.Module, pool2: nn.Module):
        super(CascadeCornerPooling, self).__init__()
        
        self.conv_up = Convolution(in_channels, out_channels)
        self.conv_up2 = Convolution(in_channels, out_channels)

        self.pool1 = pool1()
        
        self.conv_down = Convolution(in_channels, out_channels)
        self.conv_down2 = Convolution(in_channels, out_channels)
        
        self.p_conv1 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.p_conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(out_channels)
        self.p_bn2   = nn.BatchNorm2d(out_channels)
        
        self.pool2 = pool2()
        
        self.p_conv1 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(out_channels)
        
        self.conv1 = nn.Conv2d(out_channels, out_channels, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = Convolution(out_channels, out_channels)
        
    def forward(self, x):
        up1 = self.conv_up(x)
        up1 = self.pool1(up1)
        
        down1 = self.conv_down(x)
        
        merge1 = up1 + down1
        merge1 = self.p_conv1(merge1)
        merge1 = self.p_bn1(merge1)
        merge1 = self.pool2(merge1)
        
        up2 = self.conv_up2(x)
        up2 = self.pool1(up2)
        
        down2 = self.conv_down2(x)
        
        merge2 = up2 + down2
        merge2 = self.p_conv2(merge2)
        merge2 = self.p_bn2(merge2)
        merge2 = self.pool2(merge2)
        
        p_conv1 = self.p_conv1(merge1 + merge2)
        p_bn1   = self.p_bn1(p_conv1)
        
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)
        
        conv2 = self.conv2(relu1)
        return conv2

        
class CascadeTLCornerPooling(CascadeCornerPooling):
    def __init__(self, in_channels: int, out_channels: int):
        super(CascadeTLCornerPooling, self).__init__(in_channels, out_channels, LeftPool, TopPool)
        
class CascadeBRCornerPooling(CascadeCornerPooling):
    def __init__(self, in_channels: int, out_channels: int):
        super(CascadeBRCornerPooling, self).__init__(in_channels, out_channels, RightPool, BottomPool)
