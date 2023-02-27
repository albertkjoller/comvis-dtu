import numpy as np


def Pi(ph: np.ndarray) -> np.ndarray:
    """
    Maps from homogeneous coordinates to inhomogeneous coordinates
    by dividing the coordinates with the scaling factor.

    N = number of points, D = dimension.

    Args:
        ph (np.ndarray):
            (D+1, N)-dimensional coordinates given in homogeneous coordinates.

    Returns:
        np.ndarray:
            (D, N)-dimensional coordinates, now given in inhomogeneous coordinates.
    """
    return ph[:-1] / ph[-1]


def PiInv(p: np.ndarray) -> np.ndarray:
    """
    Maps from inhomogeneous coordinates to homogeneous coordinates
    by adding an extra dimension with a scaling factor of 1.

    N = number of points, D = dimension.

    Args:
        p (np.ndarray):
            (D, N)-dimensional coordinates given in inhomogeneous coordinates.

    Returns:
        np.ndarray:
            (D+1, N)-dimensional coordinates, now given in homogeneous coordinates.
    """
    _, N = p.shape
    return np.vstack([p, np.ones(N)])


def normalize2d(points: np.ndarray) -> np.ndarray:
    """
    Computes the normalization transformation, T, that normalizes the input
    points used when estimating a homography.

    Args:
        points (np.ndarray): (2, N)-dimensional points, given in inhomogeneous coordinates.

    Returns:
        np.ndarray: Normalization transformation matrix, T.
    """
    # Compute mean and standard deviation of the two axes
    mu_x, mu_y = points.mean(axis=1)
    sigma_x, sigma_y = points.std(axis=1)

    # Returns transformation matrix
    return np.array(
        [
            [1 / sigma_x, 0, -mu_x / sigma_x],
            [0, 1 / sigma_y, -mu_y / sigma_y],
            [0, 0, 1],
        ]
    )
