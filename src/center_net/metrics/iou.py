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


class AP:
    def __init__(self, min_score: float = 0.1, smooth: float = 1e-6):
        self.min_score = min_score
        self.smooth = smooth

    def perform_iou(self, y_true, y_hat):
        inter = (y_true & y_hat).sum((0, 1))
        union = (y_true | y_hat).sum((0, 1))

        return (inter + self.smooth) / (union + self.smooth)
    
    def calculare_ap_fd(self, y_true: np.array, y_hat: np.array, ious: np.array,
                        thresholds: list = [0.05, 0.5, 0.75, 0.95]) -> dict:
        """Calculate the average precision and false discovery rate for a given threshold
        
        Args:
            y_true (np.array): Ground truth of shape (N, 5) | (tlx, tly, brx, bry, class)
            y_hat (np.array): Calculated values of shape (M, 8) | (tlx, tly, brx, bry, cx, cy, score, class)
            ious (np.array): IoU values of shape (N, M)
            thresholds (list, optional): List of thresholds to calculate the AP and FD. Defaults to [0.05, 0.5, 0.75, 0.95].
            
        Returns:
            dict: Dictionary containing the AP and FD for each threshold
        """
        
        dic = {}
        
        for threshold in thresholds:
            fd = np.ones(y_hat.shape[0])
            
            iou = np.where(ious > threshold, ious, 0)
            
            best_index = np.argmax(iou, axis=1)
            ap = np.mean(iou[np.arange(y_true.shape[0]), best_index])
            fd[best_index] = 0
            
            dic["AP_{}".format(threshold)] = ap
            dic["FD_{}".format(threshold)] = np.mean(fd)
            
        return dic

    def calculate(self, y_true: np.array, y_hat: np.array, 
                  thresholds: list = [0.05, 0.5, 0.75, 0.95]) -> tuple:
        """Calculate the average precision and false discovery rate

        Args:
            y_true (np.array): Ground truth of shape (N, 5) | (tlx, tly, brx, bry, class)
            y_hat (np.array): Calculated values of shape (M, 8) | (tlx, tly, brx, bry, cx, cy, score, class)

        Returns:
            tuple: Average precision and false discovery rate
        """

        if not y_hat.size:
            return 0, {}

        y_hat = y_hat[y_hat[:, 6] > self.min_score]
        y_true = y_true[~np.all(y_true == 0, axis=1)]

        ap = 0

        iou = np.zeros((y_true.shape[0], y_hat.shape[0]))
        for i in range(y_true.shape[0]):
            for j in range(y_hat.shape[0]):
                if y_true[i, 4] == y_hat[j, 7]:
                    iou[i, j] = self.perform_iou(
                        y_true[i, :4].astype("int32").reshape(1, 4),
                        y_hat[j, :4].astype("int32").reshape(1, 4),
                    )

        if not np.any(iou):
            return 0, {}

        best_index = np.argmax(iou, axis=1)
        ap = np.mean(iou[np.arange(y_true.shape[0]), best_index])

        return ap, self.calculare_ap_fd(y_true, y_hat, iou, thresholds)
