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
