import cv2
import numpy as np


def load_cv2_image(impath: str) -> np.ndarray:
    """
    Loads an image using the opencv-module.

    Args:
        impath (str): Path to the image to be loaded.

    Returns:
        np.ndarray: A numpy array containing the loaded image with reordered color channels (for aligning with matplotlib ordering).
    """
    # Load image
    im = cv2.imread(impath)
    # Reorder channels
    return im[:, :, ::-1]
