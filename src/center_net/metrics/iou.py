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

    def pre_process(self):
        pass

    def perform_calculation(self):
        y_true, y_hat = self.y_true, self.y_hat

        x_1, y_1 = max(y_true[0], y_hat[0]), max(y_true[1], y_hat[1])
        x_2, y_2 = min(y_true[2], y_hat[2]), min(y_true[3], y_hat[3])

        interection_area = max(0, x_2 - x_1 + 1) * max(0, y_2 - y_1 + 1)

        y_true_area = (y_true[2] - y_true[0] + 1) * (y_true[3] - y_true[1] + 1)
        y_hat_area = (y_hat[2] - y_hat[0] + 1) * (y_hat[3] - y_hat[1] + 1)

        iou = interection_area / float(y_true_area + y_hat_area - interection_area)

        return iou
