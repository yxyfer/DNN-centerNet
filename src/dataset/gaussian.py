import numpy as np


def gaussian_2d(shape: tuple, sigma: float = 1) -> np.ndarray:
    """Generate gaussian map.
    Args:
        shape (tuple): Shape of the gaussian map (Height, width).
        sigma (float, optional): Sigma of the gaussian. Defaults to 1.
        
    Returns:
        np.ndarray: Gaussian map of shape (Height, width).
    """
    
    m, n = (shape[0] - 1.) / 2., (shape[1] - 1.) / 2.
    x, y = np.meshgrid(np.linspace(-n, n, shape[1]), np.linspace(-m, m, shape[0]))
    
    h = np.exp(-(x**2 + y**2) / (2*sigma**2))
    return h


def draw_gaussian(heatmap: np.array, center: tuple, radius: float, k: int = 1, delta: int = 6):
    """"Add a 2D gaussian function to a given heatmap.
    
    Args:
        heatmap (ndarray): The heatmap to which the gaussian function will be added.
        center (tuple of int): The (x, y) coordinates of the center of the gaussian function.
        radius (float): The radius of the gaussian function.
        k (float): The scaling factor for the gaussian function.
        delta (float): The standard deviation of the gaussian function.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / delta)

    x, y = center

    left, right = max(x - radius, 0), min(x + radius + 1, heatmap.shape[1])
    top, bottom = max(y - radius, 0), min(y + radius + 1, heatmap.shape[0])

    heatmap[top:bottom, left:right] = np.maximum(heatmap[top:bottom, left:right], gaussian[:bottom-top, :right-left] * k)


def gaussian_radius(det_size: tuple, min_overlap: float) -> float:
    """Get radius of gaussian.

    Args:
        det_size (tuple): (Height, Width)
        min_overlap (float): Minimum overlap. Value between 0 and 1.

    Returns:
        Float: Radius of gaussian.
    """
    height, width = det_size

    sum_hw = height + width
    mul_hw = height * width

    r1 = (sum_hw - np.sqrt(sum_hw ** 2 - 4 * mul_hw * (1 - min_overlap) / (1 + min_overlap))) / 2

    r2 = (sum_hw - np.sqrt(sum_hw ** 2 - 4 * mul_hw * (1 - min_overlap))) / 4

    r3 = (-sum_hw * min_overlap + np.sqrt((sum_hw * min_overlap) ** 2 - 4 * mul_hw * (min_overlap - 1) * min_overlap)) / (4 * min_overlap)

    return min(r1, r2, r3)