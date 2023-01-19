import matplotlib.pyplot as plt
import numpy as np
import cv2


colors = ["blue", "green", "cyan", "red", "yellow", "magenta", "peru", "azure", "slateblue", "plum"]
colors = [plt.cm.colors.to_rgb(c) for c in colors]

def display_bbox(image: np.array, bbox: np.array, print_result: bool = False,
                 filter: bool = True, min_score: float = 0.3):
    """Plot the bounding boxes on the image.

    Args:
        image (np.array): Image to plot.
        bbox (np.array): Bounding boxes to plot.
        print_result (bool, optional): Print the result?. Defaults to False.
        filter (bool, optional): Filter the bounding boxes?. Defaults to True.
        min_score (float, optional): Minimum score to filter. Defaults to 0.2.
    """

    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    thick = int((300 + 300) // 900)
    
    for b in bbox:
        x1, y1, x2, y2 = b[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = colors[int(b[-1])]
         
        if filter and b[6] < min_score:
            continue
        
        if print_result:
            print(f"bbox: {x1, y1, x2, y2}, class: {int(b[-1])}, score: {b[6]}")
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thick)
        cv2.putText(image, str(int(b[-1])), (x1, y1 - 3), 0, 1e-3 * 300, color, thick//3)
    
    plt.figure(figsize=(8, 8)) 
    plt.imshow(image)
    plt.show()