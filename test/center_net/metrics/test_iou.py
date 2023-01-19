from src.center_net.metrics import IoU
import numpy as np


class TestIoU:
    def test_full(self):
        box = np.array([0, 0, 1, 1, -1, -1])
        assert IoU(box, box) == 1.0

    def test_half(self):
        box_true = np.array([0, 10, 0, 10, -1, -1])
        box_hat = np.array([0, 12, 0, 12, -1, -1])
        assert IoU(box_true, box_hat)() == 0.5714285867346933
