"""
This file is part of the "Pose Calib" project.
It is subject to the license terms in the LICENSE file found
in the top-level directory of this distribution.

@author Pavel Rojtberg
"""

import cv2
import numpy as np

from distvis import make_distort_map

def project_img(img, sz, K, rvec, t, flags=cv2.INTER_LINEAR):
    """
    projects a 2D object (image) according to parameters
    @param img: image to project
    @param sz: size of the final image  
    """
    # construct homography
    R = cv2.Rodrigues(rvec)[0]
    H = K.dot(np.array([R[:, 0], R[:, 1], t]).T)
    H /= H[2, 2]

    return cv2.warpPerspective(img, H, sz, flags=flags)

class BoardPreview:
    SIZE = (640, 480)
    
    def __init__(self, img):
        # generate styled board image
        self.img = img
        self.img = cv2.flip(self.img, 0)  # flipped when printing
        self.img[self.img == 0] = 64  # set black to gray
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        self.img[:, :, 0::2] = 0  # set red and blue to zero

        self.shadow = np.ones(self.img.shape[:2], dtype=np.uint8)  # used for overlap score
    
    def create_maps(self, K, cdist, sz):
        if cdist is None:
            cdist = np.array([0., 0., 0., 0.])
        
        self.sz = sz
        scale = np.diag((self.SIZE[0]/sz[0], self.SIZE[1]/sz[1], 1))
        K = scale.dot(K)
        
        sz = self.SIZE
        self.Knew = cv2.getOptimalNewCameraMatrix(K, cdist, sz, 1)[0]
        self.maps = make_distort_map(K, sz, cdist, self.Knew)

    def project(self, r, t, shadow=False, inter=cv2.INTER_NEAREST):
        img = project_img(self.shadow if shadow else self.img, self.SIZE, self.Knew, r, t)
        img = cv2.remap(img, self.maps[0], self.maps[1], inter)
        img = cv2.resize(img, self.sz, interpolation=inter)
        return img