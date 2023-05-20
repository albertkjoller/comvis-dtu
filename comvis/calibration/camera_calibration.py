from typing import Union

import numpy as np

from ..geometry.planes import hest
from ..utils.coordinates import Pi, PiInv
from ..utils.operators import CrossOp

def estimateHomographies(
        Q_omega: np.array,
        qs: Union[np.array, list[np.array], tuple[np.array]]) -> list[np.array]:
    assert Q_omega.shape[0] == 3, f"""Q_omega must be 3D imhomogenous points
            but has shape: {Q_omega.shape}"""

    # Construct Q tilde.
    Q_tilde = Q_omega.copy()
    Q_tilde[-1] = 1

    # Estimate homographies.
    Hs = [hest(qi, Q_tilde) for qi in qs]

    return Hs

def estimate_b(Hs: list[np.array]) -> np.array:

    def get_v(H: np.array, alpha: int, beta: int) -> np.array:
        v = np.array([
            H[0, alpha] * H[0, beta],
            H[0, alpha] * H[1, beta] + H[1, alpha] * H[0, beta],
            H[1, alpha] * H[1, beta],
            H[2, alpha] * H[0, beta] + H[0, alpha] * H[2, beta],
            H[2, alpha] * H[1, beta] + H[1, alpha] * H[2, beta],
            H[2, alpha] * H[2, beta]
        ])
        return v

    V = np.vstack([np.array([get_v(H, 0, 1), get_v(H, 0, 0) - get_v(H, 1, 1)]) for H in Hs])

    u, s, v = np.linalg.svd(V)
    b = v[-1]
    return b / np.linalg.norm(b)

def estimateIntrinsics(Hs: list[np.array]) -> np.array:
    """Takes the output of estimateHomographies."""
    # b = [B11 B12 B22 B13 B23 B33]
    b = estimate_b(Hs)

    # Use Zhang et al appendix B
    deltay = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1]**2)
    lambda_ = b[5] - (b[3]**2 + deltay * (b[1] * b[3] - b[0] * b[4])) / b[0]
    f = np.sqrt(lambda_ / b[0])
    alphaf = np.sqrt(lambda_ * b[0] / (b[0] * b[2] - b[1]**2))
    betaf = - b[1] * f**2 * alphaf / lambda_
    deltax = betaf * deltay / alphaf - b[3] * f**2 / lambda_
    alpha = alphaf / f
    beta = betaf / f

    K = np.array([
        [f, beta*f, deltax],
        [0, alpha*f, deltay],
        [0, 0, 1]
    ])
    return K

def estimateExtrinsics(
        K: np.array,
        Hs: list[np.array]
    ) -> tuple[list[np.array], list[np.array]]:

    def estimateRt(
            H: np.array,
            Kinv: np.array
        ) -> tuple[np.array]:
        lambdai = (np.linalg.norm(Kinv @ H[:, 0]) + np.linalg.norm(Kinv @ H[:, 1])) / 2
        lambdai_inv = 1 / lambdai
        t = lambdai_inv * Kinv @ H[:, 2]

        # Correct for flipped solutions, i.e., behind the camera.
        if t[-1] < -1e-7:
            return estimateRt(-H, Kinv)
        else:
            r0 = lambdai_inv * Kinv @ H[:, 0]
            r1 = lambdai_inv * Kinv @ H[:, 1]
            r2 = np.cross(r0, r1)
            R = np.array([r0, r1, r2]).T
        return R, t

    Kinv = np.linalg.inv(K)
    Rs, ts = list(), list()
    for H in Hs:
        R, t = estimateRt(H, Kinv)
        Rs.append(R)
        ts.append(t)

    return Rs, ts

def calibratecamera(
        qs: Union[np.array, list[np.array], tuple[np.array]],
        Q: np.array
    ) -> tuple[np.array]:
    """qs is a list of homogenous points and
    Q is a 3xn array 3D imhomogenous points."""
    Hs = estimateHomographies(Q, qs)
    K = estimateIntrinsics(Hs)
    Rs, ts = estimateExtrinsics(K, Hs)
    return K, Rs, ts

def pest(Q, q):
    _, num_points = q.shape

    Bs = []
    for i in range(num_points):
        Q_i = Q[:, i]
        q_i = PiInv(q[:, i].reshape(-1, 1)).flatten()
        q_i_x = CrossOp(q_i)

        B_i = np.kron(Q_i, q_i_x)
        Bs.extend(B_i)

    B = np.vstack([Bs])

    _, _, Vh = np.linalg.svd(B)
    V = Vh.T

    # Estimate homography
    P_est = V[:, -1].reshape(4, 3).T

    # normalizing factor
    frob_norm = np.linalg.norm(P_est, 'fro')
    #print(frob_norm)
    assert np.allclose(frob_norm, 1), "Something is wrong..."
    return P_est
