from typing import Union, Tuple

import numpy as np


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
