"""
This file is part of the "Pose Calib" project.
It is subject to the license terms in the LICENSE file found
in the top-level directory of this distribution.

@author Pavel Rojtberg
"""

import cv2

import numpy as np
import numpy.linalg as la

def get_bounds(thresh, mask):
    MAX_OVERLAP = 0.9
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    # look for the largest object that is not masked
    while contours:
        mx = np.argmax([len(c) for c in contours])
        contour = contours[mx]
        aabb = cv2.boundingRect(contour)

        x, y, w, h = aabb
        if mask is not None and (cv2.countNonZero(mask[y:y + h, x:x + w]) / (w * h) > MAX_OVERLAP):
            del contours[mx]  # remove from candidates
            continue

        return (aabb, contour)

    return None

def make_distort_map(K, sz, dist, Knew):
    """
    creates a map for distorting an image as a opposed to the default
    behaviour of undistorting
    @param sz: width, height
    """
    pts = np.array(np.meshgrid(range(sz[0]), range(sz[1]))).T.reshape(-1, 1, 2)
    dpts = cv2.undistortPoints(pts.astype(np.float32), K, dist, P=Knew)

    return dpts.reshape(sz[0], sz[1], 2).T

def sparse_undistort_map(K, sz, dist, Knew, step=1):
    """
    same output as initUndistortRectifyMap, but sparse
    @param sz: width, height
    @return: distorted points, original points
    """
    zero = np.zeros(3)
    pts = np.array(np.meshgrid(range(0, sz[0], step), range(0, sz[1], step))).T.reshape(-1, 1, 2)

    if step == 1:
        dpts = cv2.initUndistortRectifyMap(K, dist, None, Knew, sz, cv2.CV_32FC2)[0].transpose(1, 0, 2)
    else:
        pts3d = cv2.undistortPoints(pts.astype(np.float32), Knew, None)
        pts3d = cv2.convertPointsToHomogeneous(pts3d).reshape(-1, 3)
        dpts = cv2.projectPoints(pts3d, zero, zero, K, dist)[0]

    shape = (sz[0] // step, sz[1] // step, 2)

    return dpts.reshape(-1, 2).reshape(shape), pts.reshape(shape)

def get_diff_heatmap(img1, img2, colormap=True):
    """
    creates a heatmap from two point images
    """
    sz = img1.shape[:2]
    l2diff = la.norm((img1 - img2).reshape(-1, 2), axis=1).reshape(sz).T

    if colormap:
        l2diff = cv2.normalize(l2diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        l2diff = cv2.applyColorMap(l2diff, cv2.COLORMAP_JET)

    return l2diff, l2diff.max()

def loc_from_dist(pts, dpts, mask=None, lower=False, thres=1.0):
    """
    compute location based on distortion strength
    @param pts: sampling locations
    @param dpts: distorted points
    @param mask: mask for ignoring locations
    @param lower: find location with minimal distortion instead
    @param thres: distortion strength to use as threshold [%]
    """
    diff = la.norm((pts - dpts).reshape(-1, 2), axis=1)
    diff = diff.reshape(pts.shape[0:2]).T
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    bounds = None

    while not bounds and thres >= 0 and thres <= 1:
        if lower:
            thres += 0.05
            thres_img = cv2.threshold(diff, thres * 255, 255, cv2.THRESH_BINARY_INV)[1]
        else:
            thres -= 0.05
            thres_img = cv2.threshold(diff, thres * 255, 255, cv2.THRESH_BINARY)[1]

        bounds = get_bounds(thres_img, mask)

        if bounds is None:
            continue

        # ensure area is not 0
        if bounds[0][2] * bounds[0][3] == 0:
            bounds = None
    
    if bounds is None:
        return None, None

    return np.array(bounds[0]), thres_img
