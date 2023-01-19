import numpy as np
import argparse
import torch
from PIL import Image

from src.center_net.keypoints import decode, filter_detections, rescale_detection
from src.center_net.network import CenterNet
from src.helpers import display_bbox


def decode_ouputs(outputs: list, K: int = 70, num_dets: int = 1000, n: int = 5) -> np.array:
    """From outputs of the model, decode the detections and centers

    Args:
        outputs (list): Outputs of the model
        K (int, optional): Number of centers. Defaults to 70.
        num_dets (int, optional): Number of detections. Defaults to 100.
        n (int, optional): Odd number 3 or 5. Determines the scale of the central region. Defaults to 5.
    Returns:
        np.array: Decoded detections and centers (bbox, class)
    """
     
    detections, centers = decode(*outputs, K, 3, 0.5, num_dets=num_dets)
    detections = filter_detections(detections[0], centers[0], n=n)
    detections = rescale_detection(detections)
        
    return detections


def pred(model, image):
    model.eval()
    with torch.no_grad():
        return model(image)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image")
    parser.add_argument(
        "--nfilter", help="To not filter the bbox. Default False", action="store_false"
    )
    parser.add_argument(
        "--minscore", default=0.1, type=float, help="Minimum score for the bbox. Default 0.1"
    )
    parser.add_argument(
        "--center", action="store_true", help="To display the centers. Default False"
    )

    args = parser.parse_args()

    model = CenterNet()
    model.load_state_dict(torch.load("models/center_net_model.pth"))
    
    image = np.array(Image.open(args.image), dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    output = pred(model, torch.from_numpy(image).unsqueeze(0).float())
    detections = decode_ouputs(output)

    img = image.transpose((1, 2, 0))

    display_bbox(img, detections,
                 filter=args.nfilter, min_score=args.minscore,
                 plot_center=args.center)
