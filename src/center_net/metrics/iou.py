import numpy as np

from src.helpers.bbox import get_bbox


class IoU(object):
    """This is an IoU calculator

    It takes in the standard output y_hat from the program and the truth value
    y_true and perform an IoU (intersection over Union) on the data.

    It pre-processes it to make the data understandable and then performs the
    IoU and returns it.
    Args:
        y_true (_type_): _description_
        y_hat  (_type_): _description_
    """

    def get_bounding_box(self, coords):
        """Returns the fours coords of a bounding box

        Args:
            coords (np.array): contains the top-left and bottom-right corner
            points.
        """
        x1, y1, x2, y2 = coords

    def __init__(self, y_true, y_hat):
        self.y_true = y_true
        self.y_hat = y_hat

        self.pre_process()
        self._iou = self.perform_calculation()

    def __call__(self):
        return self._iou

    def __eq__(self, other):
        return self._iou == other

    def __ne__(self, other):
        return self._iou != other

    def get_values(self, box):
        # return np.array([[box[0], box[1]], [box[2], box[3]]])
        return np.array([get_bbox(box)[:4]])

    def pre_process(self):
        self.y_hat = self.get_values(self.y_hat)
        self.y_true = self.get_values(self.y_true)

    def intersection(self):
        return (self.y_true & self.y_hat).sum((0, 1))

    def union(self):
        return (self.y_true | self.y_hat).sum((0, 1))

    def perform_calculation(self):
        SMOOTH = 1e-6
        return (self.intersection() + SMOOTH) / (self.union() + SMOOTH)
