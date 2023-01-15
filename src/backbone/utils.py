import torch
from torch import nn
from .network_backbone import BackboneNetwork

def load_backbone_model(model_path: str, remove_last_layer: int = 2) -> nn.Sequential:
    """Load a pretrained model.
    
    Args:
        model_path (str): Path to the pretrained model.
        remove_last_layer (int): Number of layer to remove. Defaults to 2.
    
    Returns:
        nn.Sequential: Loaded model.
    """
    
    model = BackboneNetwork()
    model.load_state_dict(torch.load(model_path))
    
    return nn.Sequential(*list(model.network.children())[:-remove_last_layer])