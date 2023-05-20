import numpy as np

from ..utils.coordinates import Pi, PiInv, normalize2d
from ..utils.operators import CrossOp

import cv2
from tqdm import tqdm

def hest(q1: np.ndarray, q2: np.ndarray, normalize: bool = False) -> np.ndarray:
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
        T1, T2 = normalize2d(Pi(q1)), normalize2d(Pi(q2))
        q1, q2 = T1 @ q1, T2 @ q2

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


def estHomographyRANSAC(kp1, des1, kp2, des2, seed=0):

    np.random.seed(seed)

    matcher = cv2.BFMatcher_create(crossCheck=True)
    matches = matcher.match(des1, des2)

    Hest_required_points = 4
    compute_dist = lambda Hest, p1, p2: np.linalg.norm(Pi((Hest @ p2)) - Pi(p1), ord=2, axis=0) + np.linalg.norm(Pi((np.linalg.inv(Hest) @ p1)) - Pi(p2), ord=2, axis=0)

    sigma = 3.0
    n_iter = 200

    # For finding inliers and consensus
    threshold = 3.84 * sigma**2

    # Get match coordinates
    num_points = len(matches)
    p1, p2 = np.zeros((3, num_points)), np.zeros((3, num_points))

    for i, match in enumerate(matches):
        # getting coordinates
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        # points
        p1[:, i] = np.array([[x1, y1, 1]])
        p2[:, i] = np.array([[x2, y2, 1]])


    max_num_inliers = -np.inf
    for _ in tqdm(range(n_iter)):

        # GET RANDOM POINTS
        rand_matches = np.random.choice(matches, Hest_required_points, replace=False)
        p1_rand, p2_rand = np.zeros((3, 8)), np.zeros((3, 8))
        for i, match in enumerate(rand_matches):
            # getting coordinates
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt
            # points
            p1_rand[:, i] = np.array([x1, y1, 1])
            p2_rand[:, i] = np.array([x2, y2, 1])

        # INITIAL Homography estimate
        Hest = hest(p1_rand, p2_rand)

        # Compute distance and determine inliers
        dists = compute_dist(Hest, p1, p2)
        inliers = dists < threshold
        consensus = inliers.sum()

        if consensus > max_num_inliers:
            max_num_inliers = consensus
            bestInliers = inliers

    best_Hest = hest(p1[:, bestInliers], p2[:, bestInliers])
    return best_Hest, matches, bestInliers, p1, p2
