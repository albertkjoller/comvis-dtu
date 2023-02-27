import numpy as np


def compute_signed_dist(line: np.ndarray, points: np.ndarray) -> float:
    """
    Computes the signed distance between the line and the inputted points.

    Args:
        line (np.ndarray): A line given in homogeneous vector form, i.e. ax + by + c = 0 --> [a, b, c]^T
        points (np.ndarray): (D+1, N)-dimensional points (homogeneous coordinates).

    Returns:
        float: (N,)-dimensional array of distances between input points and the normalized line.
    """

    line_dist_from_origo = round((l[:-1] ** 2).sum(), 2)

    # Line tangent to unit circle?
    if line_dist_from_origo != 1.0:
        # Normalize if not
        line = line / np.linalg.norm(line[:-1])

    # Last element
    pw = points[-1, :]

    return abs(line @ points) / (pw * np.linalg.norm(line[:-1]))
