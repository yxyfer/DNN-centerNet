import numpy as np


class AP:
    def __init__(self, min_score: float = 0):
        """Initialize the AP/IoU metric

        Args:
            min_score (float, optional): Minimum score to consider a bounding box. Defaults to 0.
        """

        self.min_score = min_score

    def perform_iou(self, y_true, y_hat):
        inter = (y_true & y_hat).sum((0, 1))
        union = (y_true | y_hat).sum((0, 1))

        return (inter + self.smooth) / (union + self.smooth)

    def caculate_ap_fd(
        self,
        y_true: np.array,
        y_hat: np.array,
        ious: np.array,
        thresholds: list = [0.05, 0.5, 0.75, 0.95],
    ) -> dict:
        """Calculate the average precision and false discovery rate for given thresholds

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

            y_true
            ap = np.mean(iou[np.arange(y_true.shape[0]), best_index])
            fd[best_index] = 0

            dic["AP_{}".format(threshold)] = ap
            dic["FD_{}".format(threshold)] = np.mean(fd)

        return dic

    def calculate(
        self,
        y_true: np.array,
        y_hat: np.array,
        thresholds: list = [0.05, 0.5, 0.75, 0.95],
    ) -> tuple:
        """Calculate the average IoU, average precision and false discovery rate

        Args:
            y_true (np.array): Ground truth of shape (N, 5) | (tlx, tly, brx, bry, class)
            y_hat (np.array): Calculated values of shape (M, 8) | (tlx, tly, brx, bry, cx, cy, score, class)
            thresholds (list, optional): List of thresholds to calculate the AP and FD. Defaults to [0.05, 0.5, 0.75, 0.95].
            keep_zeros (bool, optional): Keep zeros in the average precision calculation. Defaults to False.

        Returns:
            tuple: Average IoU, average precision and false discovery rate
        """

        if not y_hat.size:
            return 0, {}

        y_hat = y_hat[y_hat[:, 6] > self.min_score]
        y_true = y_true[~np.all(y_true == 0, axis=1)]

        iou = np.zeros((y_true.shape[0], y_hat.shape[0]))
        for i in range(y_true.shape[0]):
            for j in range(y_hat.shape[0]):
                if y_true[i, 4] == y_hat[j, 7]:
                    iou[i, j] = self.iou(y_true[i, :4], y_hat[j, :4])

        if not np.any(iou):
            return 0, {}

        best_indexes = np.argmax(iou, axis=1)
        avg_iou = np.mean(iou[np.arange(y_true.shape[0]), best_indexes])

        return np.mean(avg_iou), self.caculate_ap_fd(y_true, y_hat, iou, thresholds)
