from src.center_net.network import CenterNet
from src.center_net.keypoints import decode, filter_detections, rescale_detection
from src.helpers import display_bbox, get_image, get_bbox
from src.center_net.metrics.iou import AP

import torch
import numpy as np
from PIL import Image


from os import listdir
from os.path import isfile, join


class NotebookHelpers:
    def __init__(self, base_path: str, model_path: str = "models/center_net_model.pth"):
        self.model_path = model_path

        self.base_path = base_path
        self.imgs_base_path = base_path + "images/"

        self.thresholds = ["0.05", "0.5", "0.75", "0.95"]
        self.metrics = ["AP_", "FD_"]

        self._load_model()
        self._get_metric_thresholds()
        self._get_img_indexes()

    def print_metric_thresholds(self, curr_metric_thresholds: dict):
        print("Average Precision & False Discoveries per threshold:")
        for metric_threshold in self.metric_thresholds:
            print(
                f"{metric_threshold:7}: {round(curr_metric_thresholds[metric_threshold], 3)}"
            )
        print("\n")

    def get_ap_dataset(self, imgs: np.array = None, print: bool = False):
        DICT_INDEX = 1

        if not imgs:
            imgs = self.img_indexes

        ap_fd_res = dict(
            (metric_threshold, []) for metric_threshold in self.metric_thresholds
        )
        for image in imgs:
            curr_metrics = self.pred_image(image)
            for metric_threhsold in self.metric_thresholds:
                ap_fd_res[metric_threhsold].append(
                    curr_metrics[DICT_INDEX][metric_threhsold]
                )

        metric_thresholds_avg = self._get_metrics_average(ap_fd_res)

        if print:
            self.print_metric_thresholds(metric_thresholds_avg)
        else:
            return metric_thresholds_avg

    def pred_image(
        self, image_num: str, display: bool = False, display_labels: bool = False
    ):
        image_name = self.base_path + "/images/" + str(image_num) + ".png"
        label_name = self.base_path + "/labels/" + str(image_num) + ".txt"

        image = get_image(image_name)

        detections = self._predict(torch.from_numpy(image).unsqueeze(0).float())
        labels = self._get_labels(label_name)

        metrics = AP()
        aIoU, metric_thresholds_res = metrics.calculate(labels, detections)

        if display:
            if display_labels:
                print(labels + "\n")
            print(f"average IoU {round(aIoU, 3)}\n")
            self.print_metric_thresholds(metric_thresholds_res)

            img = image.transpose((1, 2, 0))
            display_bbox(img, detections, False, min_global_score=0.1, plot_center=True)
        else:
            return aIoU, metric_thresholds_res

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

    def _get_metric_thresholds(self):
        metric_thresholds = []

        for threshold in self.thresholds:
            for metric in self.metrics:
                metric_thresholds.append(metric + threshold)

        self.metric_thresholds = metric_thresholds

    def _get_metrics_average(self, ap_fd_res: dict):
        avg_metrics = dict(
            (metric_threshold, 0) for metric_threshold in self.metric_thresholds
        )
        for metric_threshold in self.metric_thresholds:
            avg_metrics[metric_threshold] = np.mean(ap_fd_res[metric_threshold])

        return avg_metrics

    def _get_img_indexes(self):
        img_paths = [
            img_path
            for img_path in listdir(self.imgs_base_path + "")
            if isfile(join(self.imgs_base_path, img_path))
        ]

        img_indexes = []
        for img_path in img_paths:
            if img_path == ".DS_Store":
                continue
            img_indexes.append(img_path.removesuffix(".png"))

        self.img_indexes = img_indexes
