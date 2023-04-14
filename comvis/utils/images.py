from typing import List

import numpy as np
import cv2

from .coordinates import Pi, PiInv

def get_warped_corners(im, H):
    # Find the corners of the image
    h, w = im.shape[:2]
    corners = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).T
    # Warp corners
    warped_corners = Pi(H @ PiInv(corners))

    xmin = int(np.floor(np.min(warped_corners.T[:, 0])))
    ymin = int(np.floor(np.min(warped_corners.T[:, 1])))

    xmax = int(np.ceil(np.max(warped_corners.T[:, 0])))
    ymax = int(np.ceil(np.max(warped_corners.T[:, 1])))
    return np.array([[xmin, xmax], [ymin, ymax]])

def get_ranges(imgs, Hs):

    all_corners = []
    for i, im in enumerate(imgs):
        all_corners.append(get_warped_corners(im, Hs[i]).flatten())

    all_corners = []
    for i, im in enumerate(imgs):
        all_corners.append(get_warped_corners(im, Hs[i]).flatten())

    all_corners = np.vstack(all_corners)
    xmin, ymin = all_corners.min(axis=0)[::2]
    xmax, ymax = all_corners.max(axis=0)[1::2]

    return [xmin, xmax], [ymin, ymax]

def warpImage(im, H, xRange, yRange):
    T = np.eye(3)
    T[:2, 2] = [-xRange[0], -yRange[0]]
    H = T@H
    outSize = (xRange[1]-xRange[0], yRange[1]-yRange[0])
    mask = np.ones(im.shape[:2], dtype=np.uint8)*255
    imWarp = cv2.warpPerspective(im, H, outSize)
    maskWarp = cv2.warpPerspective(mask, H, outSize)
    return imWarp, maskWarp

def warpImagesAuto(imgs: List, Hs: List):
    xRange, yRange = get_ranges(imgs, Hs)

    warped_imgs, warped_masks = list(zip(*[warpImage(im, H, xRange, yRange) for im, H in list(zip(imgs, Hs))]))
    return warped_imgs, warped_masks

def mix_warped_images(im1, im1_mask, im2, im2_mask):
    mixed_image = im2.copy()
    mixed_image[im2_mask == 0] = im1[im2_mask == 0]
    return mixed_image