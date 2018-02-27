import cv2


class ImageTransformer:
    def __init__(self, config):
        self.img_size = (config.img_shape[1], config.img_shape[0])
        self.M = cv2.getPerspectiveTransform(config.warp_src, config.warp_dst)
        self.Minv = cv2.getPerspectiveTransform(config.warp_dst, config.warp_src)
        self.mtx = config.undistort_mtx
        self.dist = config.undistort_dist

    def warp(self, img):
        return cv2.warpPerspective(img, self.M, self.img_size, flags=cv2.INTER_NEAREST)

    def unwarp(self, img):
        return cv2.warpPerspective(img, self.Minv, self.img_size, flags=cv2.INTER_NEAREST)

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)



