import numpy as np
import scipy
import itertools as it
from ..utils.coordinates import Pi, PiInv, normalize2d

from typing import List


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


def triangulate_nonlin(
    pixel_coords: List[np.ndarray], projection_matrices: List[np.ndarray]
) -> np.ndarray:
    """
    Triangulation using non-linear optimziation (squared error).

    Args:
        pixel_coords (list): A list of tuples representing pixel coordinates of the point in each image. Each tuple should contain two values: the x-coordinate and the y-coordinate.
        projection_matrices (list): A list of 3x4 projection matrices for each image. Each projection matrix should be a numpy array of shape (3, 4).

    Returns:
        numpy.ndarray: A numpy array of shape (4,) representing the 3D point in homogeneous coordinates.
    """

    def compute_residuals(Q: np.ndarray) -> np.ndarray:
        Q = Q.reshape(-1, 1)
        return np.vstack(
            [
                Pi(projection_matrices[i] @ Q) - pixel_coords[i]
                for i in range(len(pixel_coords))
            ]
        ).flatten()

    x0 = triangulate(pixel_coords=pixel_coords, projection_matrices=projection_matrices)
    res = scipy.optimize.least_squares(compute_residuals, x0)
    return res.x


def checkerboard_points(n, m):
    """
    Returns checkerboard points aranged in a nxm matrix.

    Args:
        n (int): rows
        m (int): columns

    Returns:
        np.array: checkerboard points
    """
    Q = [[i - (n-1)/2, j - (m-1)/2, 0] for i, j in it.product(range(n), range(m))]
    return np.vstack(Q).T


def SampsonsDistance(Fest: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    """_summary_

    Args:
        Fest (np.ndarray): Estimated fundamental matrix
        p1 (np.ndarray): (2+1, 1) point from one image given in homogeneous coordinates.
        p2 (np.ndarray): Corresponding (2+1, 1) point from another image given in homogeneous coordinates.
    """
    q1, q2 = p1, p2

    q2_Fest         = (q2.reshape(1, -1) @ Fest).reshape(-1, 1)
    # q2_Fest         = PiInv(Pi(q2_Fest))
    Fest_q1         = (Fest @ q1.reshape(-1, 1)).reshape(-1, 1)
    # Fest_q1         = PiInv(Pi(Fest_q1))

    numerator       = (q2.reshape(1, -1) @ Fest @ q1.reshape(-1, 1))**2
    denominator     = q2_Fest[0]**2 + q2_Fest[1]**2 + Fest_q1[0]**2 + Fest_q1[1]**2

    return numerator / denominator