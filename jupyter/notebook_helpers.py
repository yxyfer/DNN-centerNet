from src.center_net.network import CenterNet
from src.center_net.keypoints import decode, filter_detections, rescale_detection
from src.helpers import display_bbox, get_image, get_bbox
from src.center_net.metrics.iou import AP

import torch
import numpy as np
from PIL import Image


class NotebookHelpers:
    def __init__(self, base: str, model_path: str = "models/center_net_model.pth"):
        self.base = base
        self.model_path = model_path

        self._load_model()

    def get_ap_dataset(self, images: np.array):
        ap, fd = [], []
        for image in images:
            curr_ap, curr_fd = self.pred_image(image)
            ap.append(curr_ap)
            fd.append(curr_fd)

        return np.mean(ap), np.mean(fd)

    def pred_image(self, image_num: str, display: bool = False):
        image_name = self.base + "/images/" + str(image_num) + ".png"
        label_name = self.base + "/labels/" + str(image_num) + ".txt"

        image = get_image(image_name)

        detections = self._predict(torch.from_numpy(image).unsqueeze(0).float())
        labels = self._get_labels(label_name)

        ap = AP()
        ap, fd = ap.calculate(labels, detections)

        if display:
            print(labels)
            print(ap)
            print(fd)

            img = image.transpose((1, 2, 0))
            display_bbox(img, detections, False, min_global_score=0.1, plot_center=True)

        return ap, fd

    def _get_labels(self, label_name: str):
        f = open(label_name)
        labels = f.read().splitlines()
        f.close()

        return np.roll(
            np.array([list(map(int, label.split(","))) for label in labels[1:]]),
            -1,
            axis=1,
        )

    def _predict(self, image, K=70, num_dets=1000, n=5):
        self.model.eval()
        with torch.no_grad():
            outs = self.model(image)[0]

            detections, centers = decode(*outs, K, 3, 0.5, num_dets=num_dets)

            detections = filter_detections(detections[0], centers[0], n=n)

            detections = rescale_detection(detections)

            return np.array(detections)

    def _load_model(self):
        self.model = CenterNet()
        self.model.load_state_dict(torch.load(self.model_path))
