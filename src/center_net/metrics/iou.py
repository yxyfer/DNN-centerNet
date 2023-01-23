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


class AP:
    def __init__(self, min_score: float = 0.1, smooth: float = 1e-6):
        self.min_score = min_score
        self.smooth = smooth
        
    def perform_iou(self, y_true, y_hat):
        inter = (y_true & y_hat).sum((0, 1))
        union = (y_true | y_hat).sum((0, 1))
    
        return (inter + self.smooth) / (union + self.smooth)
        
    def calculate(self, y_true: np.array, y_hat: np.array) -> tuple:
        """Calculate the average precision and false discovery rate

        Args:
            y_true (np.array): Ground truth of shape (N, 5) | (tlx, tly, brx, bry, class)
            y_hat (np.array): Calculated values of shape (M, 8) | (tlx, tly, brx, bry, cx, cy, score, class)
            
        Returns:
            tuple: Average precision and false discovery rate
        """
        
        y_hat = y_hat[y_hat[:, 6] > self.min_score]
        y_true = y_true[~np.all(y_true == 0, axis=1)]

        fd = np.ones(y_hat.shape[0])
        ap = 0
        
        for i in range(y_true.shape[0]):
            best_iou = 0
            best_index = None
            
            for j in range(y_hat.shape[0]):
                if y_true[i, 4] == y_hat[j, 7]:
                    iou = self.perform_iou(y_true[i, :4].astype('int32').reshape(1, 4),
                                           y_hat[j, :4].astype('int32').reshape(1, 4))
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_index = j
                        
            ap += best_iou
            if best_index is not None:
                fd[best_index] = 0
                
        if y_hat.shape[0] == 0:
            fd_rate = 0
        else:
            fd_rate = fd.sum() / y_hat.shape[0]
            
        return ap / y_true.shape[0], fd_rate
    