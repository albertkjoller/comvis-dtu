import numpy as np

from ..utils.coordinates import Pi, PiInv, normalize2d
from ..utils.operators import CrossOp


def hest(q1: np.ndarray, q2: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Estimates the homography from points obtained from two different images.
    Input is homogeneous 2D-coordinates given in each of the cameras image planes.

    Args:
        q1 (np.ndarray): (2+1, N)-dimensional 2D coordinates from image plane 1, given in homogeneous coordinates.
        q2 (np.ndarray): (2+1, N)-dimensional 2D coordinates from image plane 2, given in homogeneous coordinates.
        normalize (bool, optional): Whether to normalize the points or not. Defaults to True.

    Returns:
        np.ndarray: Estimated homography. A matrix of dimension (3,3).
    """
    if normalize:
        # Get normalization matrices and normalize points)
        T1, T2 = normalize2d(q1), normalize2d(q2)
        q1, q2 = Pi(T1 @ PiInv(q1)), Pi(T2 @ PiInv(q2))

    # Setup B-matrix
    B = np.vstack(
        [np.kron(q2[:, i].reshape(1, -1), CrossOp(q1[:, i])) for i in range(len(q2.T))]
    )

    # Compute SVD
    _, _, Vh = np.linalg.svd(B)
    V = Vh.T

    # Estimate homography - take
    H_est = V[:, -1].reshape(3, 3).T

    # Verify that the estimate is close to being correct - Frobenius norm of approx. 1.0.
    frob_norm = np.linalg.norm(H_est, "fro")
    assert np.allclose(frob_norm, 1), "Something is wrong..."

    # Either normalized or non-normalized homography estimate
    return np.linalg.inv(T1) @ H_est @ T2 if normalize else H_est

def Fest_8point(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Estimates the fundamental matrix from points obtained from two different images.
    Input is homogeneous 2D-coordinates given in each of the cameras image planes.

    Args:
        q1 (np.ndarray): (2+1, N)-dimensional 2D coordinates from image plane 1, given in homogeneous coordinates.
        q2 (np.ndarray): (2+1, N)-dimensional 2D coordinates from image plane 2, given in homogeneous coordinates.

    Returns:
        np.ndarray: Estimated fundamental matrix. A matrix of dimension (3,3).
    """
    n_points = len(q1.T)

    # Setup B matrix
    B = np.vstack([np.outer(q2[:, i], q1[:, i]).flatten() for i in range(n_points)])

    # Compute SVD
    _, _, Vh = np.linalg.svd(B)
    V = Vh.T

    # Estimate homography - take
    F_est = V[:, -1].reshape(3, 3)

    # Verify that the estimate is close to being correct - Frobenius norm of approx. 1.0.
    frob_norm = np.linalg.norm(F_est, "fro")
    assert np.allclose(frob_norm, 1), "Something is wrong..."

    # Normalize
    return F_est
