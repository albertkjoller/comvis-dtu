from typing import Union, List

import numpy as np


def CrossOp(p: np.ndarray) -> np.ndarray:
    """
    Implements the cross-operator - a matrix form of a 2-dimensional vector in homogeneous coordinates such that the cross-product to another matrix/vector can be computed as a dot-product.

    Args:
        p (np.ndarray): (3, 1)-dimensional vector of a 2D-point given in homogeneous coordinates.

    Returns:
        np.ndarray: Input vector represented in its cross-form - shape = (3, 3).
    """
    p = p.flatten()
    return np.array(
        [
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0],
        ]
    )
