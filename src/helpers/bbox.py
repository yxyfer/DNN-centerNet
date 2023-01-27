import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


colors = [
    "blue",
    "green",
    "cyan",
    "red",
    "yellow",
    "magenta",
    "peru",
    "azure",
    "slateblue",
    "plum",
]
colors = [plt.cm.colors.to_rgb(c) for c in colors]


def get_bbox(bbox):
    if len(bbox) > 5:
        x1, y1, x2, y2, cx, cy = bbox[:6]
        x1, y1, x2, y2, cx, cy = int(x1), int(y1), int(x2), int(y2), int(cx), int(cy)

        return x1, y1, x2, y2, cx, cy
    else:
        x1, y1, x2, y2 = bbox
        return x1, y1, x2, y2, -1, -1


def get_image(image_path: str):
    raw_image = np.array(Image.open(image_path), dtype=np.float32)
    normalised_image = raw_image / 255.0
    expended_image = np.expand_dims(normalised_image, axis=0)

    return expended_image


def display_bbox(
    image: np.array,
    bbox: np.array,
    print_result: bool = False,
    plot_center: bool = False,
    min_global_score: float = 0.1,
):
    """Plot the bounding boxes on the image.

    Args:
        image (np.array): Image to plot.
        bbox (np.array): Bounding boxes to plot.
        print_result (bool, optional): Print the result?. Defaults to False.
        plot_center (bool, optional): Plot the center of the bounding box?. Defaults to False.
        min_global_score (float, optional): Minimum score to filter. Defaults to 0.1.
    """

    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    thick = int((300 + 300) // 900)

    for b in bbox:
        x1, y1, x2, y2, cx, cy = get_bbox(b)
        color = colors[int(b[-1])]

        if b[6] < min_global_score:
            continue

        if print_result:
            print(f"bbox: {x1, y1, x2, y2}, class: {int(b[-1])}, score: {b[6]}")

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thick)
        cv2.putText(
            image, str(int(b[-1])), (x1, y1 - 3), 0, 1e-3 * 300, color, thick // 3
        )

        if plot_center:
            cv2.circle(image, (cx, cy), 2, color, thickness=-1)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.show()
