import cv2
import pickle
import glob
from moviepy.editor import VideoFileClip

from Config import Config
from ImageTransformer import ImageTransformer
from Thresholder import Thresholder as ts
from SlidingLaneFinder import SlidingLaneFinder
from Line import Line
from ResultVisualizer import ResultVisualizer as rv
import numpy as np


def save_image(img, filename):
    if len(img.shape) > 2:
        cv2.imwrite(filename,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img = np.dstack((img, img, img)) * 255
        cv2.imwrite(filename, img)


def load_image(file_name):
    return cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)


def process_images():
    dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
    config = Config()
    config.undistort_mtx = dist_pickle['mtx']
    config.undistort_dist = dist_pickle['dist']
    it = ImageTransformer(config)

    debug = True

    img = load_image('camera_cal/calibration1.jpg')
    save_image(it.undistort(img), 'output_images/calibration1.jpg')

    for file_name in glob.glob('test_images/*.jpg'):
        print(file_name)
        img = load_image(file_name)

        left_line = Line()
        right_line = Line()

        undist = it.undistort(img)
        if debug:
            save_image(undist, 'output_images/' + file_name + '_0.jpg')

        bitmap = ts.pipeline(undist)
        if debug:
            save_image(bitmap, 'output_images/' + file_name + '_1.jpg')

        warped_bitmap_img = it.warp(bitmap)
        if debug:
            undist_copy = undist.copy()
            undist_warped = it.warp(undist_copy)
            cv2.polylines(undist_copy, np.int32([config.warp_src]), True, (255, 0, 0), 5)
            cv2.polylines(undist_warped, np.int32([config.warp_dst]), True, (0, 255, 0), 5)

            save_image(undist_copy, 'output_images/' + file_name + '_2_1.jpg')
            save_image(undist_warped, 'output_images/' + file_name + '_2_2.jpg')
            save_image(warped_bitmap_img, 'output_images/' + file_name + '_2_3.jpg')

        if debug:
            out_sliding = SlidingLaneFinder.find(warped_bitmap_img, left_line, right_line, debug)
            save_image(out_sliding, 'output_images/' + file_name + '_3.jpg')
        else:
            SlidingLaneFinder.find(warped_bitmap_img, left_line, right_line, debug)

        color_lines_img = rv.draw_lane(warped_bitmap_img, left_line, right_line)
        if debug:
            save_image(color_lines_img, 'output_images/' + file_name + '_4.jpg')

        unwarped_color_lines_img = it.unwarp(color_lines_img)
        if debug:
            save_image(unwarped_color_lines_img, 'output_images/' + file_name + '_5.jpg')

        result = cv2.addWeighted(undist, 1, unwarped_color_lines_img, 0.3, 0)
        if debug:
            save_image(result, 'output_images/' + file_name + '_6.jpg')

        rv.stamp_radius(result, left_line, right_line)
        rv.stamp_offset(result, left_line, right_line)
        if debug:
            save_image(result, 'output_images/' + file_name + '_7.jpg')
        else:
            save_image(result, 'output_images/' + file_name)


def process_video():
    dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
    config = Config()
    config.undistort_mtx = dist_pickle['mtx']
    config.undistort_dist = dist_pickle['dist']
    it = ImageTransformer(config)

    def image_pipeline(img, left_line, right_line):
        undist = it.undistort(img)
        bitmap = ts.pipeline(undist)
        warped_bitmap_img = it.warp(bitmap)
        SlidingLaneFinder.find(warped_bitmap_img, left_line, right_line)
        color_lines_img = rv.draw_lane(warped_bitmap_img, left_line, right_line)
        unwarped_color_lines_img = it.unwarp(color_lines_img)
        result = cv2.addWeighted(undist, 1, unwarped_color_lines_img, 0.3, 0)
        rv.stamp_radius(result, left_line, right_line)
        rv.stamp_offset(result, left_line, right_line)
        return result

    for input_path in glob.glob('project_video.mp4'):
        left_line = Line()
        right_line = Line()
        in_clip = VideoFileClip(input_path)
        out_clip = in_clip.fl_image(lambda img: image_pipeline(img, left_line, right_line))
        out_clip.write_videofile('test_videos_output/' + input_path, audio=False)


process_images()
