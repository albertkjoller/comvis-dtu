import numpy as np
from ..utils.operators import CrossOp


def compute_essential_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Computes the essential matrix between two cameras.

    Args:
        R (np.ndarray): (3, 3)-dimensional rotation matrix defining the
            relative rotation between the two cameras.
        t (np.ndarray): (3, 1)-dimensional translation vector defining the
            relative translation between the two cameras.

    Returns:
        np.ndarray: The essential matrix between the two cameras.
    """
    return CrossOp(t) @ R


def compute_fundamental_matrix(
    K1: np.ndarray, K2: np.ndarray, E: np.ndarray
) -> np.ndarray:
    """_summary_

    Args:
        K1 (np.ndarray): Camera matrix of Camera1.
        K2 (np.ndarray): Camera matrix of Camera2.
        E (np.ndarray): Essential matrix computed with the relative rotation matrix
            and translation vector between Camera1 and Camera2.

    Returns:
        np.ndarray: The fundamental matrix between the two cameras.
    """
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
