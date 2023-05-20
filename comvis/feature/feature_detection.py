import numpy as np
import cv2

from typing import List

# Exercise 6.1
def gaussian1DKernel(sigma):
    # Not important further away
    std_from_the_mean = 3
    x = np.arange(-std_from_the_mean*sigma, std_from_the_mean*sigma + 1) # plus one, for symmetry of the gaussian.

    # Compute Gaussian
    g = 1 / np.sqrt(2*np.pi*sigma**2) * np.exp((-x**2) / (2*sigma**2))
    # Compute derivative of Gaussian
    gd = -x / (sigma**2) * g

    return (g.reshape(2*sigma*std_from_the_mean + 1, 1),
            gd.reshape(2*sigma*std_from_the_mean + 1, 1))

# Exercise 6.2
def gaussianSmoothing(im, sigma):
    g, gd = gaussian1DKernel(sigma)

    # I: Gaussian Smoothed image (i.e. smoothed in both directions)
    I = cv2.filter2D(cv2.filter2D(im, -1, g), -1, g.T)

    # Ix, Iy, smoothed derivatives if image.
    # Smoothed in one direction, then derivative in opposite direction
    Ix = cv2.filter2D(cv2.filter2D(im, -1, g), -1, gd.T)
    Iy = cv2.filter2D(cv2.filter2D(im, -1, g.T), -1, gd)

    return I, Ix, Iy

def smoothedHessian(im, sigma, epsilon):
    # Compute gradients
    I, Ix, Iy = gaussianSmoothing(im, sigma)
    # Get 1D Gaussian kernel
    g_eps, gd_eps = gaussian1DKernel(epsilon)

    # Compute Gaussian smoothing
    C_11, _, _ = gaussianSmoothing((Ix*Ix), epsilon)
    C_22, _, _ = gaussianSmoothing((Iy*Iy), epsilon)
    C_offdiag, _, _ = gaussianSmoothing((Ix*Iy), epsilon)

    C = np.array([[C_11, C_offdiag],
                  [C_offdiag, C_22]])

    return C

def harrisMeasure(im, sigma, epsilon, k):
    # Compute smoothed Hessian
    C = smoothedHessian(img, sigma, epsilon)
    # Extract numbers
    a, b, c = C[0,0], C[1,1], C[0,1]

    # Compute Harris metric
    return a * b - c**2 - k * (a + b)**2

def cornerDetector(im, sigma, epsilon, k, tau):
    # Compute Harris metric
    r = harrisMeasure(img, sigma, epsilon, k)

    corners = []
    # Check all pixels
    for x in range(1, r.shape[0] - 1):
        for y in range(1, r.shape[1] - 1):
            # Use acceptance criterion
            if r[x,y] > tau:
                # Use NMS (non-maximum suppresion)
                if ((r[x, y] > r[x+1, y])
                and (r[x, y] > r[x-1, y])
                and (r[x, y] > r[x, y+1])
                and (r[x, y] > r[x, y-1])):
                    corners.append([x,y])
    # Return corners
    return np.array(corners)





# Exercise 8.1
def scaleSpaced(im: np.ndarray, sigma, n) -> List[np.ndarray]:
    """_summary_

    Args:
        im (np.ndarray): Input image
        sigma (_type_):
        n (_type_):

    Returns:
        List[np.ndarray]: A scale-space pyramid of n+1 images (since input image is included).
    """
    h, w = im.shape

    scale_pyramids = [im]
    for i in range(n):
        im_ = scale_pyramids[i]
        I, Ix, Iy = gaussianSmoothing(im_, sigma * 2**i)
        scale_pyramids.append(I)

    return np.array(scale_pyramids)

# Exercise 8.2
def differenceOfGaussians(im, sigma, n) -> List[np.ndarray]:
    im_scales = scaleSpaced(im, sigma, n)
    return np.array([im_scales[i] - im_scales[i+1] for i in range(1, im_scales.__len__() - 1)])

# Exercise 8.3
def detectBlobs(im, sigma, n, tau):

    DoG = [dog.T for dog in differenceOfGaussians(im, sigma, n)]
    MaxDoG = [cv2.dilate(abs(dog), np.ones((3,3))) for dog in DoG]

    BLOBs = {}
    for i in range(n-1):
        blob_ = []
        h, w = DoG[i].shape
        for x in range(1, h - 1):
            for y in range(1, w - 1):

                if abs(DoG[i][x, y]) > tau:

                    if (
                        (DoG[i][x, y] > DoG[i][x+1, y])
                        and (DoG[i][x, y] > DoG[i][x-1, y])

                        and (DoG[i][x, y] > DoG[i][x-1, y+1])
                        and (DoG[i][x, y] > DoG[i][x,   y+1])
                        and (DoG[i][x, y] > DoG[i][x+1, y+1])

                        and (DoG[i][x, y] > DoG[i][x-1, y-1])
                        and (DoG[i][x, y] > DoG[i][x,   y-1])
                        and (DoG[i][x, y] > DoG[i][x+1, y-1])
                    ):

                        if i == 0 and DoG[0][x, y] > MaxDoG[1][x, y]:
                            blob_.append([x, y])
                        elif i == (n-1) and DoG[-1][x,y] > MaxDoG[-2][x, y]:
                            blob_.append([x, y])
                        elif i > 0 and i < (n-1) and DoG[i][x, y] > MaxDoG[i-1][x, y] and DoG[i][x, y] > MaxDoG[i+1][x, y]:
                            blob_.append([x, y])
                        else:
                            pass

        BLOBs[sigma * 2**i] = blob_
    return BLOBs
