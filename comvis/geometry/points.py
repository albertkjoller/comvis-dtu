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


def triangulate(pixel_coords, projection_matrices):
    """_summary_

    Author: ChatGPT

    Args:
        pixel_coords (list): A list of tuples representing pixel coordinates of the point in each image. Each tuple should contain two values: the x-coordinate and the y-coordinate.
        projection_matrices (list): A list of 3x4 projection matrices for each image. Each projection matrix should be a numpy array of shape (3, 4).

    Returns:
        numpy.ndarray: A numpy array of shape (4,) representing the 3D point in homogeneous coordinates.
    """

    n = len(pixel_coords)
    B = np.zeros((2 * n, 4))

    for i in range(n):
        P = projection_matrices[i]
        x, y = pixel_coords[i]
        # See slide 27 for def.
        B[2 * i] = x * P[2] - P[0]  # First row of block i in B
        B[2 * i + 1] = y * P[2] - P[1]  # Second row of block i in B

    # Solve the linear system Bx=0 using SVD (where x is the 3D point in homogeneous coordinates)
    _, _, V = np.linalg.svd(B)

    # Extract last row of V and normalize to get X
    return V[-1][:4] / V[-1][3]
