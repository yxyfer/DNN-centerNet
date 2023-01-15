from torch import nn
from ..helpers import Convolution
from .cascade_corner_pooling import CascadeTLCornerPooling, CascadeBRCornerPooling
from .center_pooling import CenterPooling

class CornerNet(nn.Module):
    def __init__(self, backbone: nn.Module):
        # -- TODO: Check all the dimensions of the network
        
        super(CornerNet, self).__init__()
        
        self.pre = nn.Sequential(
            Convolution(1, 32, stride=2),
            Convolution(32, 64, stride=2),
        )
        
        self.backbone = backbone()
        
        self.conv1 = Convolution(64, 128)
        
        self.top_left_pool = CascadeTLCornerPooling()
        self.bottom_left_pool = CascadeBRCornerPooling()
        self.center_pool = CenterPooling()
        
        self.top_left_hmap = Convolution() 
        
        