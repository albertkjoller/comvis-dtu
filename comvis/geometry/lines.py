import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from tqdm import tqdm

from ..geometry.points import SampsonsDistance
from ..geometry.planes import Fest_8point


def compute_signed_dist(line: np.ndarray, points: np.ndarray) -> float:
    """
    Computes the signed distance between the line and the inputted points.

    Args:
        line (np.ndarray): A line given in homogeneous vector form, i.e. ax + by + c = 0 --> [a, b, c]^T
        points (np.ndarray): (D+1, N)-dimensional points (homogeneous coordinates).

    Returns:
        float: (N,)-dimensional array of distances between input points and the normalized line.
    """

    line_dist_from_origo = round((line[:-1] ** 2).sum(), 2)

    # Line tangent to unit circle?
    if line_dist_from_origo != 1.0:
        # Normalize if not
        line = line / np.linalg.norm(line[:-1])

    # Last element
    pw = points[-1, :]

    return abs(line @ points) / (pw * np.linalg.norm(line[:-1]))

### FROM WEEK 7
def DrawLine(l, shape):
    #Checks where the line intersects the four sides of the image
    # and finds the two intersections that are within the frame
    def in_frame(l_im):
        q = np.cross(l.flatten(), l_im)
        q = q[:2]/q[2]
        if all(q>=0) and all(q+1<=shape[1::-1]):
            return q
    lines = [[1, 0, 0], [0, 1, 0], [1, 0, 1-shape[1]], [0, 1, 1-shape[0]]]
    P = [in_frame(l_im) for l_im in lines if in_frame(l_im) is not None]

    if (len(P)==0):
        print("Line is completely outside image")

    plt.plot(*np.array(P).T, color='r', alpha=0.5)




### RANSAC (week 7) ###

def inlier_detection(l: np.ndarray, points: np.ndarray, threshold: Union[float, int]) -> np.ndarray:

    dists = compute_signed_dist(l, points)
    return dists < threshold

def calculate_consensus(l: np.ndarray, points: np.ndarray, threshold: Union[float, int]) -> int:

    inliers = inlier_detection(l, points, threshold)
    return sum(inliers)

def drawNpoints(points: np.ndarray, n_points: int = 2):
    point_idxs = np.random.randint(0, points.shape[1], size=n_points)
    return points[:, point_idxs]

def fit_line(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Estimate line from two homogeneous coordinates.

    Args:
        p1 (np.ndarray): A homogeneous coordinate.
        p2 (np.ndarray): Another homogeneous coordinate.

    Returns:
        np.ndarray: Hold... The line!
    """

    return np.cross(p1, p2)

def test_points(n_in, n_out):
    a = (np.random.rand(n_in)-.5)*10
    b = np.vstack((a, a*.5+np.random.randn(n_in)*.25))
    points = np.hstack((b, 2*np.random.randn(2, n_out)))
    return np.random.permutation(points.T).T

def calc_N_hat(p, eps_hat, df: int) -> int:
    """_summary_

    Args:
        p (_type_): Probability
        eps_hat (_type_): Current estimate
        df (int): Degrees of freedom

    Returns:
        (int): New estimate of number of points.
    """

    return np.log(1-p) / np.log((1-(1-eps_hat)**df))

def RANSAC(points: np.ndarray, threshold: Union[float, int], n_iter: int = 5, draw_n_points: int = 2, p=0.99, seed: int = 0) -> np.ndarray:
    """
    Implementation from week 7

    Args:
        points (np.ndarray): _description_
        threshold (Union[float, int]): _description_
        n_iter (int, optional): _description_. Defaults to 5.
        draw_n_points (int, optional): _description_. Defaults to 2.
        seed (int, optional): _description_. Defaults to 0.

    Returns:
        np.ndarray: _description_
    """

    # set seed
    np.random.seed(seed)

    n = 0
    max_num_inliers = -np.inf
    estimated_line = None

    # Adaptive running lenght
    N_hat = np.inf
    M = len(points)

    while n < N_hat:
        selected_points = drawNpoints(points, draw_n_points)
        line_ = fit_line(selected_points[:, 0], selected_points[:, 1])

        num_inliers = calculate_consensus(line_, points, threshold=threshold)

        if num_inliers > max_num_inliers:
            max_num_inliers = num_inliers
            estimated_line = line_

            inliers = inlier_detection(line_, points, threshold=threshold)

            # Check for convergence-ish
            s = calculate_consensus(l, points=points, threshold=threshold)
            eps_hat = 1 - s/M
            N_hat = calc_N_hat(p, eps_hat, 2)

        n += 1

    def pca_line(x): #assumes x is a (2 x n) array of points
        d = np.cov(x)[:, 0]
        d /= np.linalg.norm(d)
        l = [d[1], -d[0]]
        l.append(-(l@x.mean(1)))
        return l

    print(f"Estimated line (before training on inliers): {estimated_line}")
    best_line = pca_line(Pi(points[:, inliers]))

    print(f"Estimated line (after training on inliers): {estimated_line}")
    print(f"Number of inliers: {max_num_inliers}")
    print(f"Number of outliers: {points.shape[1] - max_num_inliers}")
    print(f"Threshold value: {threshold}")

    return best_line, estimated_line, inliers




### RANSAC (for Fundamental) week 9 ###

def Fest_RANSAC(matches, kp1, kp2, sigma, n_iter=200, seed=0):
    np.random.seed(seed)

    # Find inliers and consensus
    threshold = 3.84 * sigma**2

    # GET REMAINING DATA POINTS
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
        rand_matches = np.random.choice(matches, 8, replace=False)
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

        # INITIAL F ESTIMATE
        Fest = Fest_8point(p1_rand, p2_rand)

        dists = np.array([SampsonsDistance(Fest, p1[:, i], p2[:, i]) for i in range(p1.shape[1])]).flatten()
        inliers = dists < threshold
        consensus = inliers.sum()

        if consensus > max_num_inliers:
            max_num_inliers = consensus
            inliers = inliers

    best_Fest = Fest_8point(p1[:, inliers], p2[:, inliers])

    return best_Fest, inliers, p1, p2
