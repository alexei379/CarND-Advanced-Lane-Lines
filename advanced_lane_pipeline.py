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

    for input_path in glob.glob('project_video-*.mp4'):
        left_line = Line()
        right_line = Line()
        in_clip = VideoFileClip(input_path)
        out_clip = in_clip.fl_image(lambda img: image_pipeline(img, left_line, right_line))
        out_clip.write_videofile('test_videos_output/' + input_path, audio=False)

process_video()

