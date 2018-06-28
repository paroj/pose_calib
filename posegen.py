"""
This file is part of the "Pose Calib" project.
It is subject to the license terms in the LICENSE file found
in the top-level directory of this distribution.

@author Pavel Rojtberg
"""

import numpy as np
import numpy.linalg as la

import cv2

from distvis import sparse_undistort_map, loc_from_dist

def gen_bin(s, e):
    """
    generate values in range by binary subdivision
    """
    t = (s + e) / 2
    lst = [(s, t), (t, e)]

    while lst:
        s, e = lst.pop(0)
        t = (s + e) / 2
        lst.append((s, t))
        lst.append((t, e))
        yield t

def unproject(p, K, cdist, Z):
    """
    project pixel back to a 3D coordinate at depth Z
    """
    p = cv2.undistortPoints(p.reshape(-1, 1, 2), K, cdist).ravel()
    return np.array([p[0], p[1], 1]) * Z

def oribital_pose(bbox, rx, ry, Z, rz=0):
    """
    @param bbox: object bounding box. note: assumes planar object with virtual Z dimension. 
    @param rx: rotation around x axis in rad
    @param ry: rotation around y axis in rad
    @param Z: distance to camera in board lengths
    @return: rvec, tvec 
    """
    Rz = cv2.Rodrigues(np.array([0., 0., rz]))[0]
    Rx = cv2.Rodrigues(np.array([np.pi + rx, 0., 0.]))[0]  # flip by 180° so Z is up
    Ry = cv2.Rodrigues(np.array([0., ry, 0.]))[0]

    R = np.eye(4)
    R[:3, :3] = (Ry).dot(Rx).dot(Rz)

    # translate board to its center
    Tc = np.eye(4)
    Tc[3, :3] = R[:3, :3].dot(bbox * [-0.5, -0.5, 0])

    # translate board to center of image
    T = np.eye(4)
    T[3, :3] = bbox * [-0.5, -0.5, Z]

    # rotate center of board
    Rf = la.inv(Tc).dot(R).dot(Tc).dot(T)

    return cv2.Rodrigues(Rf[:3, :3])[0].ravel(), Rf[3, :3]

def pose_planar_fullscreen(K, cdist, img_size, bbox):
    KB = K.dot([bbox[0], bbox[1], 0])  # ignore principal point
    Z = (KB[0:2] / img_size).min()
    pB = KB / Z

    r = np.array([np.pi, 0, 0])  # flip image
    # move board to center, org = bl
    p = np.array([img_size[0] / 2 - pB[0] / 2, img_size[1] / 2 + pB[1] / 2])
    t = unproject(p, K, cdist, Z)
    return r, t

def pose_from_bounds(src_ext, tgt_rect, K, cdist, img_sz):
    rot90 = tgt_rect[3] > tgt_rect[2]

    MIN_WIDTH = img_sz[0] // 3.333

    if rot90:
        src_ext = src_ext.copy()
        src_ext[0], src_ext[1] = src_ext[1], src_ext[0]

        if tgt_rect[3] < MIN_WIDTH:
            scale = MIN_WIDTH / tgt_rect[2]
            tgt_rect[3] = MIN_WIDTH
            tgt_rect[2] *= scale
    else:
        if tgt_rect[2] < MIN_WIDTH:
            scale = MIN_WIDTH / tgt_rect[2]
            tgt_rect[2] = MIN_WIDTH
            tgt_rect[3] *= scale

    aspect = src_ext[0] / src_ext[1]

    # match aspect ratio of tgt to src, but keep tl
    if not rot90:
        # adapt height
        tgt_rect[3] = tgt_rect[2] / aspect
    else:
        # adapt width
        tgt_rect[2] = tgt_rect[3] * aspect

    r = np.array([np.pi, 0, 0])

    # org is bl
    if rot90:
        R = cv2.Rodrigues(r)[0]
        Rz = cv2.Rodrigues(np.array([0., 0., -np.pi / 2]))[0]
        R = R.dot(Rz)
        r = cv2.Rodrigues(R)[0].ravel()
        # org is tl

    Z = (K[0, 0] * src_ext[0]) / tgt_rect[2]

    # clip to image region
    max_off = img_sz - tgt_rect[2:4]
    tgt_rect[0:2] = tgt_rect[0:2].clip([0, 0], max_off)

    if not rot90:
        tgt_rect[1] += tgt_rect[3]

    t = unproject(np.array([tgt_rect[0], tgt_rect[1]], dtype=np.float32), K, cdist, Z)

    if not rot90:
        tgt_rect[1] -= tgt_rect[3]

    return r, t, tgt_rect

class PoseGeneratorDist:
    """
    generate poses based on min/ max distortion
    """
    SUBSAMPLE = 20

    def __init__(self, img_size):
        self.img_size = img_size

        self.stats = [1, 1]  # number of (intrinsic, distortion) poses

        self.orbitalZ = 1.6
        rz = np.pi / 8

        # valid poses:
        # r_x, r_y -> -70° .. 70°
        self.orbital = (
            gen_bin(np.array([-(70 / 180) * np.pi, 0, self.orbitalZ, rz]), np.array([(70 / 180) * np.pi, 0, self.orbitalZ, rz])),
            gen_bin(np.array([0, -(70 / 180) * np.pi, self.orbitalZ, rz]), np.array([0, (70 / 180) * np.pi, self.orbitalZ, rz]))
        )

        self.mask = np.zeros(np.array(img_size) // self.SUBSAMPLE, dtype=np.uint8).T
        self.sgn = 1

    def compute_distortion(self, K, cdist, subsample=1):
        return sparse_undistort_map(K, self.img_size, cdist, K, subsample)

    def get_pose(self, bbox, nk, tgt_param, K, cdist):
        """
        @param bbox: bounding box of the calibration pattern
        @param nk: number of keyframes captured so far
        @param tgt_param: parameter that should be optimized by the pose
        @param K, cdist: current calibration estimate
        """
        if nk == 0:
            # init sequence: first keyframe 45° tilted to camera
            return oribital_pose(bbox, 0, np.pi / 4, self.orbitalZ, np.pi / 8)
        if nk == 1:
            # init sequence: second keyframe
            return pose_planar_fullscreen(K, cdist, self.img_size, bbox)
        if tgt_param < 4:
            # orbital pose is used for focal length
            axis = (tgt_param + 1) % 2  # f_y -> r_x
            
            self.stats[0] += 1
            r, t = oribital_pose(bbox, *next(self.orbital[axis]))

            if tgt_param > 1:
                off = K[:2, 2].copy()
                off[tgt_param - 2] += self.img_size[tgt_param - 2] * 0.05 * self.sgn
                off3d = unproject(off, K, cdist, t[2])
                off3d[2] = 0
                t += off3d
                self.sgn *= -1

            return r, t

        dpts, pts = self.compute_distortion(K, cdist, self.SUBSAMPLE)

        bounds = loc_from_dist(pts, dpts, mask=self.mask)[0]
        
        if bounds is None:
            # FIXME: anything else?
            print("loc_from_dist failed. return orbital pose instead of crashing")
            return self.get_pose(bbox, nk, 3, axis, K, cdist)
        
        self.stats[1] += 1
        r, t, nbounds = pose_from_bounds(bbox, bounds * self.SUBSAMPLE, K, cdist, self.img_size)
        x, y, w, h = np.ceil(np.array(nbounds) / self.SUBSAMPLE).astype(int)
        self.mask[y:y + h, x:x + w] = 1

        return r, t


