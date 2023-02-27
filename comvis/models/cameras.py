from typing import Union, Tuple, List, Optional
import numpy as np

from ..utils.coordinates import Pi, PiInv
from .distortion import dist


def camera_matrix(
    f: Union[int, float],
    res: Tuple[int, int],
    alpha: Union[int, float] = 1.0,
    beta: Union[int, float] = 0.0,
) -> np.ndarray:
    """
    Computes the camera matrix from the focal length and the resolution.
    The slope of the pixels is given by the fraction (beta / alpha). Per default the slope is 0, meaning square pixels.

    Args:
        f (Union[int, float]): focal length of the camera.
        res (Tuple[int, int]): resolution of the image.
        alpha (Union[int, float], optional): Part of determining the slope for non-square pixels. Defaults to 1.0.
        beta (Union[int, float], optional): Part of determining the slope for non-square pixels. Defaults to 0.0.

    Returns:
        np.ndarray: Camera matrix, K.
    """
    return np.array(
        [
            [f, 0, res[0] / 2],
            [0, f, res[1] / 2],
            [0, 0, 1],
        ]
    )


def projectpoints(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    Q: np.ndarray,
    distCoeffs: Optional[Union[List[float], np.ndarray]] = None,
) -> np.ndarray:
    """
    Projects 3D inhomogeneous points to the image plane based on the setting
    defined by the camera matrix, rotation matrix and translation vector. Output will be
    2D points lying on the image plane.

    Args:
        K (np.ndarray): Camera matrix. Shape = (3, 3)
        R (np.ndarray): Rotation matrix. Shape = (3, 3)
        t (np.ndarray): Translation vector. Shape = (3, 1)
        Q (np.ndarray): (3, N)-dimensional points to be projected, given in inhomogeneous coordinates.
        distCoeffs(Union[List[float], np.ndarray], optional): Distortion coefficients of the camera. Defaults to [].

    Returns:
        np.ndarray: (2, N)-dimensional projected points, given in inhomogeneous coordinates.
    """
    # Vertical stacking of rotation matrix and translation vector
    Rt = np.hstack([R, t.reshape(-1, 1)])  # shape = (3, 4)
    # Convert points to homogeneous coordinates
    Ph = PiInv(Q)

    # Project distorted points if distCoeffs is specified
    if distCoeffs is not None:
        return K @ PiInv(dist(Pi(Rt @ Ph), distCoeffs))

    # Project points to image plane (in inhomogeneous coordinates)
    Qh = K @ Rt @ Ph
    return Pi(Qh)
