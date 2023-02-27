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
    im = im[:, :, ::-1]
    return im.astype(float) / 255


def warpImage(im: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Warps and image with the homography.

    Args:
        im (np.ndarray): Image to be warped
        H (np.ndarray): Homography.

    Returns:
        np.ndarray: Warped image.
    """
    return cv2.warpPerspective(im, H, (im.shape[1], im.shape[0]))
