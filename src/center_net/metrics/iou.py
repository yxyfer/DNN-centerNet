import numpy as np

class AP:
    def __init__(self, min_score: float = 0):
        """Initialize the AP/IoU metric

        Args:
            min_score (float, optional): Minimum score to consider a bounding box. Defaults to 0.
        """
        
        self.min_score = min_score

    def iou(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the IoU between two bounding boxes

        Args:
            y_true (np.array): True bounding box of shape (4,) | (tlx, tly, brx, bry)
            y_pred (np.array): Predicted bounding box of shape (4,) | (tlx, tly, brx, bry)

        Returns:
            float: IoU value
        """
        
        intersection_y1 = max(y_true[0], y_pred[0])
        intersection_x1 = max(y_true[1], y_pred[1])
        intersection_y2 = min(y_true[2], y_pred[2])
        intersection_x2 = min(y_true[3], y_pred[3])
        
        intersection_area = max(intersection_y2 - intersection_y1, 0) * max(intersection_x2 - intersection_x1, 0)
        
        y_true_area = (y_true[2] - y_true[0]) * (y_true[3] - y_true[1])
        
        y_pred_area = (y_pred[2] - y_pred[0]) * (y_pred[3] - y_pred[1])
        
        return intersection_area / float(y_true_area + y_pred_area - intersection_area)
    
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
            fd = np.zeros(y_hat.shape[0])
            iou = np.where(ious > threshold, 1, 0)

            best_index = np.argmax(iou, axis=1)
            ap = np.mean(iou[np.arange(y_true.shape[0]), best_index])
            fd[np.where(iou.sum(axis=0) == 0)] = 1

            dic["AA_{}".format(threshold)] = ap
            dic["FD_{}".format(threshold)] = np.mean(fd)

        return dic


    def calculate(
        self,
        y_true: np.array,
        y_hat: np.array,
        thresholds: list = [0.05, 0.5, 0.75, 0.95],
        keep_zeros: bool = False,
    ) -> tuple:
        """Calculate the average precision and false discovery rate
        Args:
            y_true (np.array): Ground truth of shape (N, 5) | (tlx, tly, brx, bry, class)
            y_hat (np.array): Calculated values of shape (M, 8) | (tlx, tly, brx, bry, cx, cy, score, class)
            thresholds (list, optional): List of thresholds to calculate the AP and FD. Defaults to [0.05, 0.5, 0.75, 0.95].
            keep_zeros (bool, optional): Keep zeros in the average precision calculation. Defaults to False.

        Returns:
            tuple: Average precision and false discovery rate
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

        best_index = np.argmax(iou, axis=1)
        m_iou = iou[np.arange(y_true.shape[0]), best_index]
        if not keep_zeros:
            m_iou = m_iou[m_iou != 0]

        m_iou = np.mean(m_iou)
        
        return m_iou, self.calculare_ap_fd(y_true, y_hat, iou, thresholds)
