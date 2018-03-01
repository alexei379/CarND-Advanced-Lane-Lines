import numpy as np


class Config:
    img_shape = (720, 1280)

    # Image transformation config
    undistort_mtx = None
    undistort_dist = None

    warp_src = np.float32(
        [[(img_shape[1] / 2) - 470, img_shape[0] - 60],
         [(img_shape[1] / 2) - 72, img_shape[0] - 265],
         [(img_shape[1] / 2) + 72, img_shape[0] - 265],
         [(img_shape[1] / 2) + 470, img_shape[0] - 60]])

    warp_dst = np.float32(
        [[(img_shape[1] / 2) - 400, img_shape[0]],
         [(img_shape[1] / 2) - 400, 0],
         [(img_shape[1] / 2) + 400, 0],
         [(img_shape[1] / 2) + 400, img_shape[0]]])