from torch import nn
from ...helpers import Convolution, Residual
from .cascade_corner_pooling import CascadeTLCornerPooling, CascadeBRCornerPooling
from .center_pooling import CenterPooling
from .layers import HeatMapLayer, EmbeddingLayer, OffsetLayer
from ...backbone import load_backbone_model

class CenterNet(nn.Module):
    def __init__(self):
        super(CenterNet, self).__init__()

        self.backbone = load_backbone_model("models/backbone_model.pth")
        
        self.post = nn.Sequential(
            Convolution(64, 128),
            Residual(128, 128),
            Residual(128, 128),
            Residual(128, 128),
        )
        
        self.cascade_tl_pool = CascadeTLCornerPooling(128, 128)
        self.cascade_br_pool = CascadeBRCornerPooling(128, 128)
        self.center_pool = CenterPooling(128, 128)
        
        self.hmap_tl = HeatMapLayer(128, 128, 10)
        self.hmap_br = HeatMapLayer(128, 128, 10)
        self.hmap_ct = HeatMapLayer(128, 128, 10)
        
        self.hmap_tl.heatmap[-1].bias.data.fill_(-2.19)
        self.hmap_br.heatmap[-1].bias.data.fill_(-2.19)
        self.hmap_ct.heatmap[-1].bias.data.fill_(-2.19)
        
        self.embd_tl = EmbeddingLayer(128, 128, 1)
        self.embd_br = EmbeddingLayer(128, 128, 1)
        
        self.offset_tl = OffsetLayer(128, 128, 2)
        self.offset_br = OffsetLayer(128, 128, 2)
        self.offset_ct = OffsetLayer(128, 128, 2)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.post(x)
        
        ctl_pool = self.cascade_tl_pool(x)
        cbr_pool = self.cascade_br_pool(x)
        cc_pool = self.center_pool(x)
        
        hmap_tl = self.hmap_tl(ctl_pool)
        hmap_br = self.hmap_br(cbr_pool)
        hmap_ct = self.hmap_ct(cc_pool)
        
        embd_tl = self.embd_tl(ctl_pool)
        embd_br = self.embd_br(cbr_pool)
        
        offset_tl = self.offset_tl(ctl_pool)
        offset_br = self.offset_br(cbr_pool)
        offset_ct = self.offset_ct(cc_pool)
        
        return [[hmap_tl, hmap_br, hmap_ct, embd_tl, embd_br, offset_tl, offset_br, offset_ct]]
