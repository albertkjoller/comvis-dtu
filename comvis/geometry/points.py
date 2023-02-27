import numpy as np
import itertools as it


def box3d(n: int = 16) -> np.ndarray:
    """
    Creates a box of 3D points.

    Args:
        n (int, optional): Number of points that the box consists of. Defaults to 16.

    Returns:
        np.ndarray: Points reordered as a 3D box.
    """
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i,) * n, (j,) * n, N])))

    return np.hstack(points) / 2
