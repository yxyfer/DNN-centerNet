import numpy as np
import argparse
import torch
from PIL import Image

from src.center_net.utils import decode, center_match, _rescale_dets
from src.center_net.network import CenterNet
from src.helpers import display_bbox

def decode_ouputs(outputs: list) -> list:
    """From outputs of the model, decode the detections and centers

    Args:
        outputs (list): Outputs of the model

    Returns:
        list: Decoded detections and centers (bbox, class)
    """
    
    dets, cts = decode(*outputs, 
                       K=70,
                       ae_threshold=0.5,
                       kernel=3,
                       num_dets=1000)

    dets = dets.detach().numpy()
    cts = cts.detach().numpy()

    borders = np.array([[  0., 300.,   0., 300.]])
    ratios = np.array([[0.25, 0.25]])
    sizes = np.array([[300, 300]])

    _rescale_dets(dets, cts, ratios, borders, sizes)
    return center_match(dets[0], cts[0])

def pred(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)[0]
        return decode_ouputs(output)
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--nfilter", help="To not filter the bbox. Default False", action="store_false")
    
    args = parser.parse_args()
    image = np.array(Image.open(args.image), dtype=np.float32)
    image = image / 255.
    image = np.expand_dims(image, axis=0)

    model = CenterNet()
    model.load_state_dict(torch.load("models/center_net_model.pth"))
    
    dets, cts = pred(model, torch.from_numpy(image).unsqueeze(0).float())
    
    image = image.transpose((1, 2, 0))
     
    display_bbox(image, dets, filter=args.nfilter)