#!/usr/bin/env python3

"""
This file is part of the "Pose Calib" project.
It is subject to the license terms in the LICENSE file found
in the top-level directory of this distribution.

@author Pavel Rojtberg
"""

import cv2
import argparse
import sys

from ui import UserGuidance
from utils import ChArucoDetector
    
class UVCVideoCapture:
    def __init__(self, cfg):
        self.manual_focus = True
        self.manual_exposure = True

        imsize = (int(cfg.getNode("image_width").real()), int(cfg.getNode("image_height").real()))
        
        cam_id = 0
        if not cfg.getNode("v4l_id").empty():
            cam_id = "/dev/v4l/by-id/usb-{}-video-index0".format(cfg.getNode("v4l_id").string())
        
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, imsize[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imsize[1])
        cap.set(cv2.CAP_PROP_GAIN, 0.0)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, not self.manual_focus)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        
        val = 1 / 4 if self.manual_exposure else 3 / 4
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
        
        self.cap = cap
        
        assert self.cap.isOpened()

    def set(self, prop, val):
        self.cap.set(prop, val)
        
    def read(self):
        return self.cap.read()

def add_camera_controls(win_name, cap):
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    if cap.manual_focus:
        focus = 0
        cap.set(cv2.CAP_PROP_FOCUS, focus / 100)
        cv2.createTrackbar("Focus", win_name, focus, 100, lambda v: cap.set(cv2.CAP_PROP_FOCUS, v / 100))

    if cap.manual_exposure:
        exposure = 200
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure / 1000)
        cv2.createTrackbar("Exposure", win_name, exposure, 1000, lambda v: cap.set(cv2.CAP_PROP_EXPOSURE, v / 1000))    
    
def main():
    parser = argparse.ArgumentParser(description="Interactive camera calibration using efficient pose selection")
    parser.add_argument("-c", "--config", help="path to calibration configuration (e.g. data/calib_config.yml)")
    parser.add_argument("-o", "--outfile", help="path to calibration output (defaults to calib_<cameraId>.yml)")
    parser.add_argument("-m", "--mirror", action="store_true", help="horizontally flip the camera image for display")
    args = parser.parse_args()

    if args.config is None:
        print("falling back to "+sys.path[0]+"/data/calib_config.yml")
        args.config = sys.path[0]+"/data/calib_config.yml"

    cfg = cv2.FileStorage(args.config, cv2.FILE_STORAGE_READ)
    assert cfg.isOpened()

    calib_name = "default"
    if not cfg.getNode("v4l_id").empty():
        calib_name = cfg.getNode("v4l_id").string()

    # Video I/O
    live = cfg.getNode("images").empty()
    if live:
        cap = UVCVideoCapture(cfg)
        add_camera_controls("PoseCalib", cap)
        wait = 1
    else:
        cv2.namedWindow("PoseCalib")
        cap = cv2.VideoCapture(cfg.getNode("images").string() + "frame%0d.png", cv2.CAP_IMAGES)
        wait = 0
        assert cap.isOpened()

    tracker = ChArucoDetector(cfg)


    # user guidance
    ugui = UserGuidance(tracker, cfg.getNode("terminate_var").real())

    # runtime variables
    mirror = False
    save = False

    while True:
        force = not live  # force add frame to calibration

        status, _img = cap.read()
        if status:
            img = _img
        else:
            force = False

        tracker.detect(img)

        if save:
            save = False
            force = True

        out = img.copy()

        ugui.draw(out, mirror)

        ugui.update(force)
        
        if ugui.converged:
            if args.outfile is None:
                outfile = "calib_{}.yml".format(calib_name)
            else:
                outfile = args.outfile
            ugui.write(outfile)

        if ugui.user_info_text:
            cv2.displayOverlay("PoseCalib", ugui.user_info_text, 1000 // 30)

        cv2.imshow("PoseCalib", out)
        k = cv2.waitKey(wait)

        if k == 27:
            break
        elif k == ord('m'):
            mirror = not mirror
        elif k == ord('c'):
            save = True

if __name__ == "__main__":
    main()
