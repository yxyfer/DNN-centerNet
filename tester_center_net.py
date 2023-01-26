import numpy as np
import argparse
import torch

from src.center_net.keypoints import decode, filter_detections, rescale_detection
from src.center_net.network import CenterNet
from src.helpers import display_bbox, get_image


def decode_ouputs(
    outputs: list, K: int = 70, num_dets: int = 1000, n: int = 5
) -> np.array:
    """From outputs of the model, decode the detections and centers

    Args:
        outputs (list): Outputs of the model
        K (int, optional): Number of centers. Defaults to 70.
        num_dets (int, optional): Number of detections. Defaults to 1000.
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image")
    parser.add_argument(
        "--minscore",
        default=0.2,
        type=float,
        help="Minimum score for the bbox. Default 0.1",
    )
    parser.add_argument(
        "--center", action="store_true", help="To display the centers. Default False"
    )
    parser.add_argument("--K", default=70, type=int, help="Number of centers. Default 70")
    parser.add_argument("--num_dets", default=1000, type=int,
                        help="Number of detections. Default 1000")
    parser.add_argument("--n", default=5, type=int,
                        help="Odd number 3 or 5. Determines the scale of the central region. Default 5")
    parser.add_argument("--model", default="models/center_net_model.pth",
                        help="Path to the model. Default models/center_net_model.pth")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model = CenterNet('none')
    model.load_state_dict(torch.load(args.model))

    image = get_image(args.image)

    output = pred(model, torch.from_numpy(image).unsqueeze(0).float())
    detections = decode_ouputs(output, args.K, args.num_dets, args.n)

    img = image.transpose((1, 2, 0))

    display_bbox(
        img,
        detections,
        min_score=args.minscore,
        plot_center=args.center,
    )
