from typing import Union, List

import numpy as np
import cv2

from ..utils.coordinates import Pi, PiInv


def dist(points: np.ndarray, distCoeffs: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Distorts the points captured by the camera. For instance, for making a fish-bowl effect.

    Args:
        points (np.ndarray): (2, N)-dimensional array of 2D points, given in inhomogeneous coordinates.
        distCoeffs (Union[List[float], np.ndarray]): List of distortion coefficients to be used for
            distorting input points.

    Returns:
        np.ndarray: Distorted version of the input points, still in 2D inhomogeneous coordinates.
    """
    delta_r = lambda r, coefs: sum(
        coef * r ** (2 * (i + 1)) for i, coef in enumerate(coefs)
    )
    return points * (1 + delta_r(np.linalg.norm(points, axis=0), distCoeffs))


def undistortImage(
    im: np.ndarray, K: np.ndarray, distCoeffs: Union[List[float], np.ndarray]
) -> np.ndarray:
    """
    Undistorts the image using the equation present in slides of week 2.
    Based on the inputted distortion coefficients.

    Args:
        im (np.ndarray): Distorted input image.
        K (np.ndarray): Camera matrix.
        distCoeffs (Union[List[float], np.ndarray]): Distortion coefficients of the camera.

    Returns:
        np.ndarray: The undistorted image.
    """
    # Get pixel grid and represent as homogeneous coordinates
    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    p = np.stack((x, y, np.ones(x.shape))).reshape(3, -1)

    # Equation from slides - week2
    q = np.linalg.inv(K) @ p
    q_d = PiInv(dist(Pi(q), distCoeffs))
    p_d = K @ q_d

    # Undistorted pixel coordinates
    x_d = p_d[0].reshape(x.shape).astype(np.float32)
    y_d = p_d[1].reshape(y.shape).astype(np.float32)

    # Check that everything works
    assert (p_d[2] == 1).all(), "You did a mistake somewhere"
    return cv2.remap(im, x_d, y_d, cv2.INTER_LINEAR)
