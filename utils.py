"""
This file is part of the "Pose Calib" project.
It is subject to the license terms in the LICENSE file found
in the top-level directory of this distribution.

@author Pavel Rojtberg
"""

import time

import cv2
from cv2.aruco import Dictionary_get, CharucoBoard_create, drawAxis, interpolateCornersCharuco, detectMarkers, estimatePoseCharucoBoard

import numpy as np
import numpy.linalg as la

class ChArucoDetector:
    def __init__(self, cfg):
        # configuration
        self.board_sz = np.array([int(cfg.getNode("board_x").real()), int(cfg.getNode("board_y").real())])
        self.square_len = cfg.getNode("square_len").real()
        self.ardict = Dictionary_get(int(cfg.getNode("dictionary").real()))
        
        marker_len = cfg.getNode("marker_len").real()
        self.board = CharucoBoard_create(self.board_sz[0], self.board_sz[1], self.square_len, marker_len, self.ardict)
        self.img_size = (int(cfg.getNode("image_width").real()), int(cfg.getNode("image_height").real()))

        # per frame data
        self.N_pts = 0
        self.pose_valid = False
        self.raw_img = None
        self.pt_min_markers = int(cfg.getNode("pt_min_markers").real())

        self.intrinsic_valid = False

        # optical flow calculation
        self.last_ccorners = None
        self.last_cids = None
        # mean flow if same corners are detected in consecutive frames
        self.mean_flow = None

    def set_intrinsics(self, calib):
        self.intrinsic_valid = True
        self.K = calib.K
        self.cdist = calib.cdist

    def draw_axis(self, img):
        drawAxis(img, self.K, self.cdist, self.rvec, self.tvec, self.square_len)

    def detect_pts(self, img):
        self.corners, ids, self.rejected = detectMarkers(img, self.ardict)

        self.N_pts = 0
        self.mean_flow = None

        if ids is None or ids.size == 0:
            self.last_ccorners = None
            self.last_cids = None
            return

        res = interpolateCornersCharuco(self.corners, ids, img, self.board, minMarkers=self.pt_min_markers)
        self.N_pts, self.ccorners, self.cids = res

        if self.N_pts == 0:
            return

        if not np.array_equal(self.last_cids, self.cids):
            self.last_ccorners = self.ccorners.reshape(-1, 2)
            self.last_cids = self.cids
            return

        diff = self.last_ccorners - self.ccorners.reshape(-1, 2)
        self.mean_flow = np.mean(la.norm(diff, axis=1))
        self.last_ccorners = self.ccorners.reshape(-1, 2)
        self.last_cids = self.cids

    def detect(self, img):
        self.raw_img = img.copy()

        self.detect_pts(img)

        if self.intrinsic_valid:
            self.update_pose()

    def get_pts3d(self):
        return self.board.chessboardCorners[self.cids].reshape(-1, 3)

    def get_calib_pts(self):
        return (self.ccorners.copy(), self.get_pts3d())

    def update_pose(self):
        if self.N_pts < 4:
            self.pose_valid = False
            return

        ret = estimatePoseCharucoBoard(self.ccorners, self.cids, self.board, self.K, self.cdist)
        self.pose_valid, rvec, tvec = ret

        if not self.pose_valid:
            return

        self.rvec = rvec.ravel()
        self.tvec = tvec.ravel()

        # print(cv2.RQDecomp3x3(cv2.Rodrigues(self.rvec)[0])[0])
        # print(self.tvec)

class Calibrator:
    def __init__(self, img_size):
        self.img_size = img_size

        self.nintr = 9
        self.unknowns = None  # number of unknowns in our equation system

        # initial K matrix
        # with aspect ratio of 1 and pp at center. Focal length is empirical.
        self.Kin = cv2.getDefaultNewCameraMatrix(np.diag([1000, 1000, 1]), img_size, True)
        self.K = self.Kin.copy()

        self.cdist = None

        self.flags = cv2.CALIB_USE_LU

        # calibration data
        self.keyframes = []
        self.reperr = float("NaN")
        self.PCov = np.zeros((self.nintr, self.nintr))  # parameter covariance
        self.pose_var = np.zeros(6)

        self.disp_idx = None  # index of dispersion

    def get_intrinsics(self):
        K = self.K
        return [K[0, 0], K[1, 1], K[0, 2], K[1, 2]] + list(self.cdist.ravel())

    def calibrate(self, keyframes=None):
        flags = self.flags

        if not keyframes:
            keyframes = self.keyframes

        assert(keyframes)

        nkeyframes = len(keyframes)

        if nkeyframes <= 1:
            # restrict first calibration to K matrix parameters
            flags |= cv2.CALIB_FIX_ASPECT_RATIO

        if nkeyframes <= 1:
            # with only one frame we just estimate the focal length
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

            flags |= cv2.CALIB_ZERO_TANGENT_DIST
            flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3

        t = time.process_time()

        res = calibrateCamera(keyframes, self.img_size, flags, self.Kin)

        self.reperr, self.K, self.cdist, rvecs, tvecs, self.PCov, self.N_pts = res

        self.calib_t = time.process_time() - t

        self.pose_var = compute_pose_var(rvecs, tvecs)
        self.unknowns = self.nintr + 6 * nkeyframes

        pvar = np.diag(self.PCov)
        self.mean_extr_var = mean_extr_var(pvar[self.nintr:])

        self.disp_idx = index_of_dispersion(self.get_intrinsics(), np.diag(self.PCov)[:self.nintr])
        return self.disp_idx

def index_of_dispersion(mean, var):
    """
    computes index of dispersion:
    https://en.wikipedia.org/wiki/Index_of_dispersion
    """
    return var / [abs(v) if abs(v) > 0 else 1. for v in mean]

def mean_extr_var(var):
    """
    computes the mean of the extrinsic variances
    @param var: variance vector excluding the intrinsic parameters
    """
    assert(len(var) % 6 == 0)
    nframes = len(var) // 6
    my_var = var[:6].copy()

    for i in range(1, nframes - 1):
        my_var += var[6 * i:6 * (i + 1)]

    return my_var / nframes

def estimate_pt_std(res, d, n):
    """
    estimate the accuracy of point measurements given the reprojection error
    @param res: the reprojection error 
    """
    return res / np.sqrt(1 - d / (2 * n))

def Jc2J(Jc, N_pts, nintr=9):
    """
    decompose a compact 'single view' jacobian into a sparse 'multi view' jacobian
    @param Jc: compact single view jacobian 
    @param N_pts: number of points per view
    @param nintr: number of intrinsic parameters
    """
    total = np.sum(N_pts)

    J = np.zeros((total * 2, nintr + 6 * len(N_pts)))
    J[:, :nintr] = Jc[:, 6:]

    i = 0

    for j, n in enumerate(N_pts):
        J[2 * i:2 * i + 2 * n, nintr + 6 * j:nintr + 6 * (j + 1)] = Jc[2 * i:2 * i + 2 * n, :6]
        i += n

    return J

def compute_pose_var(rvecs, tvecs):
    ret = np.empty(6)
    reuler = np.array([cv2.RQDecomp3x3(cv2.Rodrigues(r)[0])[0] for r in rvecs])

    # workaround for the given board so r_x does not oscilate between +-180°
    reuler[:, 0] = reuler[:, 0] % 360

    ret[0:3] = np.var(reuler, axis=0)
    ret[3:6] = np.var(np.array(tvecs) / 10, axis=0).ravel()  # [mm]
    return ret

def compute_state_cov(pts3d, rvecs, tvecs, K, cdist, flags):
    """
    state covariance from current intrinsic and extrinsic estimate
    """
    P_cam = []
    N_pts = [len(pts) for pts in pts3d]

    # convert to camera coordinate system
    for i in range(len(pts3d)):
        R = cv2.Rodrigues(rvecs[i])[0]
        P_cam.extend([R.dot(P) + tvecs[i].ravel() for P in pts3d[i]])

    zero = np.array([0, 0, 0.], dtype=np.float32)

    # get jacobian
    Jc = cv2.projectPoints(np.array(P_cam), zero, zero, K, cdist)[1]
    J = Jc2J(Jc, N_pts)
    JtJ = J.T.dot(J)
    
    if flags & (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3):
        # TODO: remove the according fixed rows so we can invert this 
        return np.zeros_like(JtJ)
    
    return la.inv(JtJ)

def calibrateCamera(keyframes, img_size, flags, K):
    pts2d = []
    pts3d = []
    N = 0

    for p2d, p3d in keyframes:
        pts2d.append(p2d)
        pts3d.append(p3d)
        N += len(p2d)

    res = cv2.calibrateCamera(np.array(pts3d), np.array(pts2d), img_size, K, None, flags=flags)

    reperr, K, cdist, rvecs, tvecs = res
    cov = compute_state_cov(pts3d, rvecs, tvecs, K, cdist, flags)

    return reperr, K, cdist, rvecs, tvecs, cov, N
