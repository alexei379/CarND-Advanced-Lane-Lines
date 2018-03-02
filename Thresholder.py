import cv2
import numpy as np


class Thresholder():
    def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        # binary_output = np.copy(img) # Remove this line
        return binary_output


    @staticmethod
    def color_threshold(img, s_thresh=(0, 255), v_thresh=(0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2]
        v_binary = np.zeros_like(v_channel)
        v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

        binary_output = np.zeros_like(v_channel)
        binary_output[(s_binary == 1) & (v_binary == 1)] = 1

        return binary_output

    @staticmethod
    def pipeline(img):
        ksize = 9
        preprocessed_image = np.zeros_like(img[:, :, 0])
        gradx = Thresholder.abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(12, 255))
        grady = Thresholder.abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(25, 255))
        c_binary = Thresholder.color_threshold(img, s_thresh=(100, 255), v_thresh=(50, 255))
        preprocessed_image[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 1
        return preprocessed_image
