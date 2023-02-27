from typing import Union, Tuple

import numpy as np

from ..utils.coordinates import Pi, PiInv


def camera_matrix(f: Union[int, float], res: Tuple[int, int]) -> np.ndarray:
    """
    Computes the camera matrix from the focal length and the resolution.

    Args:
        f (Union[int, float]): focal length of the camera.
        res (Tuple[int, int]): resolution of the image.

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
    K: np.ndarray, R: np.ndarray, t: np.ndarray, Q: np.ndarray
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

    Returns:
        np.ndarray: (2, N)-dimensional projected points, given in inhomogeneous coordinates.
    """
    # Vertical stacking of rotation matrix and translation vector
    Rt = np.hstack([R, t.reshape(-1, 1)])
    # Convert points to homogeneous coordinates
    Ph = PiInv(Q)
    # Project points to image plane
    Qh = K @ Rt @ Ph
    # Map to inhomogeneous coordinates for pixel interpretation
    return Pi(Qh)
