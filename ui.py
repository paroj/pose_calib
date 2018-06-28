"""
This file is part of the "Pose Calib" project.
It is subject to the license terms in the LICENSE file found
in the top-level directory of this distribution.

@author Pavel Rojtberg
"""

import cv2
import datetime

import numpy as np

from utils import Calibrator
from render import BoardPreview
from posegen import PoseGeneratorDist

def debug_jaccard(img, tmp):
    dbg = img.copy() + tmp * 2
    cv2.imshow("jaccard", dbg * 127)

class UserGuidance:
    AX_NAMES = ("red", "green", "blue")
    INTRINSICS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
    POSE = ("rx", "ry", "rz", "tx", "ty", "tz")

    SQUARE_LEN_PIX = 12
    # parameters that are optimized by the same board poses
    PARAM_GROUPS = [(0, 1, 2, 3), (4, 5, 6, 7, 8)]

    def __init__(self, tracker, var_terminate=0.1):
        # get geometry from tracker
        self.tracker = tracker
        self.allpts = np.prod(tracker.board_sz - 1)
        self.square_len = tracker.board.getSquareLength()
        self.SQUARE_LEN_PIX = int(self.square_len)

        self.img_size = tracker.img_size

        self.overlap = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.uint8)

        # preview image
        self.board = BoardPreview(self.tracker.board.draw(tuple(tracker.board_sz * self.SQUARE_LEN_PIX)))

        self.calib = Calibrator(tracker.img_size)
        self.min_reperr_init = float("inf")

        # desired pose of board for first frame
        # translation defined in terms of board dimensions
        self.board_units = np.array([tracker.board_sz[0], tracker.board_sz[1], tracker.board_sz[0]]) * self.square_len
        self.board_warped = None

        self.var_terminate = var_terminate
        self.pconverged = np.zeros(self.calib.nintr, dtype=np.bool)

        self.converged = False
        self.tgt_param = None

        # actual user guidance
        self.pose_reached = False
        self.capture = False
        self.still = False
        self.user_info_text = ""

        self.posegen = PoseGeneratorDist(self.img_size)

        # set first pose
        self.set_next_pose()

    def calibrate(self):
        if len(self.calib.keyframes) < 2:
            # need at least 2 keyframes
            return

        pvar_prev = np.diag(self.calib.PCov)[:self.calib.nintr]
        first = len(self.calib.keyframes) == 2

        index_of_dispersion = self.calib.calibrate().copy()

        pvar = np.diag(self.calib.PCov)[:self.calib.nintr]

        if not first:
            total_var_prev = np.sum(pvar_prev)
            total_var = np.sum(pvar)

            if total_var > total_var_prev:
                # del self.calib.keyframes[-1]
                print("note: total var degraded")
                # return

            # check for convergence
            rel_pstd = 1 - np.sqrt(pvar) / np.sqrt(pvar_prev)
            # np.set_printoptions(linewidth=800)
            # print(np.abs(np.sqrt(var) / vals))
            # print(rel_pstd[self.tgt_param])
            #assert rel_pstd[self.tgt_param] >= 0, self.INTRINSICS[self.tgt_param] + " degraded"
            if rel_pstd[self.tgt_param] < 0:
                print(self.INTRINSICS[self.tgt_param] + " degraded")
            for g in self.PARAM_GROUPS:
                if self.tgt_param not in g:
                    continue

                converged = []

                for p in g:
                    # if index_of_dispersion[p] < 0.05:
                    if rel_pstd[p] > 0 and rel_pstd[p] < self.var_terminate:
                        if not self.pconverged[p]:
                            converged.append(self.INTRINSICS[p])
                            self.pconverged[p] = True

                if converged:
                    print("{} converged".format(converged))

        # print(self.calib.get_intrinsics())
        # print(index_of_dispersion)
        index_of_dispersion[self.pconverged] = 0

        self.tgt_param = index_of_dispersion.argmax()

        # how well is the requirement 5x more measurements than unknowns is fulfilled
        # print(self.N_pts*2/self.unknowns, self.N_pts, self.unknowns)
        # print("keyframes: ", len(self.keyframes))

        # print("pvar min", self.pose_var.argmin())
        # print(np.diag(K), cdist)

    def set_next_pose(self):
        nk = len(self.calib.keyframes)

        self.tgt_r, self.tgt_t = self.posegen.get_pose(self.board_units,
                                                       nk,
                                                       self.tgt_param,
                                                       self.calib.K,
                                                       self.calib.cdist)

        self.board.create_maps(self.calib.K, self.calib.cdist, self.img_size)
        self.board_warped = self.board.project(self.tgt_r, self.tgt_t)

    def pose_close_to_tgt(self):
        if not self.tracker.pose_valid:
            return False

        if self.tgt_r is None:
            return False

        self.overlap[:, :] = self.board_warped[:, :, 1] != 0

        Aa = np.sum(self.overlap)

        tmp = self.board.project(self.tracker.rvec, 
                                 self.tracker.tvec, 
                                 shadow=True)
        Ab = np.sum(tmp)
        # debug_jaccard(self.overlap, tmp)
        self.overlap *= tmp[:, :]
        Aab = np.sum(self.overlap)

        # circumvents instability during initialization and large variance in depth later on
        jaccard = Aab / (Aa + Ab - Aab)

        return jaccard > 0.85

    def update(self, force=False, dry_run=False):
        """
        @return True if a new pose was captured
        """
        if not self.calib.keyframes and self.tracker.N_pts >= self.allpts // 2:
            # try to estimate intrinsic params from single frame
            self.calib.calibrate([self.tracker.get_calib_pts()])

            if not np.isnan(self.calib.K).any() and self.calib.reperr < self.min_reperr_init:
                self.set_next_pose()  # update target pose
                self.tracker.set_intrinsics(self.calib)
                self.min_reperr_init = self.calib.reperr

        self.pose_reached = force and self.tracker.N_pts > 4

        if self.pose_close_to_tgt():
            self.pose_reached = True

        # we need at least 57.5 points after 2 frames
        # and 15 points per frame from then
        n_required = ((self.calib.nintr + 2 * 6) * 5 + 3) // (2 * 2)  # integer ceil

        if len(self.calib.keyframes) >= 2:
            n_required = 6 // 2 * 5

        self.still = self.tracker.mean_flow is not None and self.tracker.mean_flow < 2
        # use all points instead to ensure we have a stable pose
        self.pose_reached *= self.tracker.N_pts >= n_required

        self.capture = self.pose_reached and (self.still or force)

        if not self.capture:
            return False

        self.calib.keyframes.append(self.tracker.get_calib_pts())

        # update calibration with all keyframe
        self.calibrate()

        # use the updated calibration results for tracking
        self.tracker.set_intrinsics(self.calib)

        self.converged = self.pconverged.all()

        if dry_run:
            # drop last frame again
            del self.caib.keyframes[-1]

        if self.converged:
            self.tgt_r = None
        else:
            self.set_next_pose()

        self._update_user_info()

        return True

    def _update_user_info(self):
        self.user_info_text = ""

        if len(self.calib.keyframes) < 2:
            self.user_info_text = "initialization"
        elif not self.converged:
            action = None
            axis = None
            if self.tgt_param < 2:
                action = "rotate"
                # do not consider r_z as it does not add any information
                axis = self.calib.pose_var[:2].argmin()
            else:
                action = "translate"
                # do not consider t_z
                axis = self.calib.pose_var[3:6].argmin() + 3

            param = self.INTRINSICS[self.tgt_param]
            self.user_info_text = "{} '{}' to minimize '{}'".format(action, self.POSE[axis], param)
        else:
            self.user_info_text = "converged at MSE: {}".format(self.calib.reperr)

        if self.pose_reached and not self.still:
            self.user_info_text += "\nhold camera steady"

    def draw(self, img, mirror=False):
        if self.tgt_r is not None:
            img[self.board_warped != 0] = self.board_warped[self.board_warped != 0]

        if self.tracker.pose_valid:
            self.tracker.draw_axis(img)

        if mirror:
            cv2.flip(img, 1, img)

    def seed(self, imgs):
        for img in imgs:
            self.tracker.detect(img)
            self.update(force=True)

    def write(self, outfile):
        flags = [(cv2.CALIB_FIX_PRINCIPAL_POINT, "+fix_principal_point"),
                 (cv2.CALIB_ZERO_TANGENT_DIST, "+zero_tangent_dist"),
                 (cv2.CALIB_USE_LU, "+use_lu")]

        fs = cv2.FileStorage(outfile, cv2.FILE_STORAGE_WRITE)
        fs.write("calibration_time", datetime.datetime.now().strftime("%c"))
        fs.write("nr_of_frames", len(self.calib.keyframes))
        fs.write("image_width", self.calib.img_size[0])
        fs.write("image_height", self.calib.img_size[1])
        fs.write("board_width", self.tracker.board_sz[0])
        fs.write("board_height", self.tracker.board_sz[1])
        fs.write("square_size", self.square_len)

        flags_str = " ".join([s for f, s in flags if self.calib.flags & f])
        fs.writeComment("flags: " + flags_str)

        fs.write("flags", self.calib.flags)
        fs.write("fisheye_model", 0)
        fs.write("camera_matrix", self.calib.K)
        fs.write("distortion_coefficients", self.calib.cdist)
        fs.write("avg_reprojection_error", self.calib.reperr)
        fs.release()

