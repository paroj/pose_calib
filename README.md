Abstract
========

The choice of poses for camera calibration with planar patterns is only rarely considered - yet the calibration precision heavily depends on it. This work presents a pose selection method that finds a compact and robust set of calibration poses and is suitable for interactive calibration. Consequently, singular poses that would lead to an unreliable solution are avoided explicitly, while poses reducing the uncertainty of the calibration are favoured. For this, we use uncertainty propagation. Our method takes advantage of a self-identifying calibration pattern to track the camera pose in real-time. This allows to iteratively guide the user to the target poses, until the desired quality level is reached. Therefore, only a sparse set of key-frames is needed for calibration. The method is evaluated on separate training and testing sets, as well as on synthetic data. Our approach performs better than comparable solutions while requiring 30% less calibration frames.

Citing
======
If you use this application for scientific work, please consider citing us as
```
@inproceedings{rojtberg2018,
    author = "Rojtberg, Pavel and Kuijper, Arjan",
    title = "Efficient Pose Selection for Interactive Camera Calibration",
    year = "2018",
    booktitle = "Proceedings of the IEEE International Symposium for Mixed and Augmented Reality 2018 (To appear)",
    organization={IEEE}
}
```

Usage
=====

Call `pose_calib.py` with a [calibration config file](data/calib_config.yml) as

```
$ ./pose_calib.py data/calib_config.yml
```

Press `m` to toggle between normal and mirrored display. You can find the [default pattern image here](data/board.png).
After convergence, the resulting calibration properties will be written as `calib_<cameraId>.yml`.

Camera, image resolution and charuco settings can be changed via the calibration config file.

A pre-build package for Ubuntu is installable via the snapcraft store as

```
sudo snap install posecalib
```
