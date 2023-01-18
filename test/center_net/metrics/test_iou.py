from src.center_net.metrics import IoU

class TestIoU():
    def test_three(self):
        box = [0, 0, 1, 1]
        assert IoU(box, box) == 1.