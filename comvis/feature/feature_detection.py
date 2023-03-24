import numpy as np
import cv2

# Gaussian 1D kernel. 
def gaussian1DKernel(sigma):
    
    std_from_the_mean = 3
    
    x = np.arange(-std_from_the_mean*sigma, 
                  std_from_the_mean*sigma  + 1) # plus one, for symmetry of the gaussian. 
    
    g = 1 / np.sqrt(2*np.pi*sigma**2) * np.exp((-x**2) / (2*sigma**2))
    
    gd = -x / (sigma**2) * g
    
    return (g.reshape(2*sigma*std_from_the_mean + 1, 1), 
            gd.reshape(2*sigma*std_from_the_mean + 1, 1))

# Gaussian Smoothing from week 6:
def gaussianSmoothing(im, sigma):
    g, gd = gaussian1DKernel(sigma)
    
    # I: Gaussian Smoothed image (i.e. smoothed in both directions)
    I = cv2.filter2D(cv2.filter2D(im, -1, g), -1, g.T)
    
    # Ix, Iy, smoothed derivatives if image. 
    # Smoothed in one direction, then derivative in opposite direction
    Ix = cv2.filter2D(cv2.filter2D(im, -1, g), -1, gd.T)
    Iy = cv2.filter2D(cv2.filter2D(im, -1, g.T), -1, gd)
    
    return I, Ix, Iy