from torch import nn
import torch

def load_pretrained_model(path: str, model: nn.Module, nb_layer: int = -1) -> nn.Module:
    """Load a pretrained model.

    Args:
        path (str): Path to the pretrained model.
        model (nn.Module): Model to load.
        nb_layer (int): Number of layer to load. Defaults to -1 (all layers).

    Returns:
        nn.Module: Loaded model.
    """
    
    model.load_state_dict(torch.load(path))
    
    if nb_layer != -1:
        model.network = nn.Sequential(*[model.network[i] for i in range(nb_layer)])
        
    return model