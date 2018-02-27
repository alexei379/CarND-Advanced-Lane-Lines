import numpy as np
import cv2


class ResultVisualizer():
    ym_per_pix = 4.1 / 115  # meters per pixel in y dimension | 42px lane mark on gmap, 102 px = 10m => lane mark is 4.1m
    xm_per_pix = 3.7 / 615  # meters per pixel in x dimension | lane width 38px on gmap, 102 px = 10m => lane width 3.7m
    # 195 px radius, 48px = 200m => min radius 800m

    @staticmethod
    def stamp_offset(img, left_line, right_line):
        center_px = (left_line.starting_x_avg + right_line.starting_x_avg)/ 2
        center_offset_m = (center_px - img.shape[1]/2) * ResultVisualizer.xm_per_pix

        offset_text = format("Center offset: %.2f m" % center_offset_m)
        cv2.putText(img, offset_text, (0, 100), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

    @staticmethod
    def stamp_radius(img, left_line, right_line):
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ResultVisualizer.ym_per_pix, left_line.bestx * ResultVisualizer.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ResultVisualizer.ym_per_pix, right_line.bestx * ResultVisualizer.xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (
                2 * left_fit_cr[0] * y_eval * ResultVisualizer.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                2 * right_fit_cr[0] * y_eval * ResultVisualizer.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        left_curverad = left_line.add_and_get_radius(left_curverad)
        right_curverad = right_line.add_and_get_radius(right_curverad)

        # Now our radius of curvature is in meters
        avg_radius = (left_curverad + right_curverad) / 2
        radius_text = format("Radius: %d m" % avg_radius)
        cv2.putText(img, radius_text, (0, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

    @staticmethod
    def draw_lane(warped, left_line, right_line, lane_color = (0, 255, 0)):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_line.bestx = left_line.best_fit[0] * ploty ** 2 + left_line.best_fit[1] * ploty + left_line.best_fit[2]
        left_fitx = left_line.bestx.astype(int)
        right_line.bestx = right_line.best_fit[0] * ploty ** 2 + right_line.best_fit[1] * ploty + right_line.best_fit[2]
        right_fitx = right_line.bestx.astype(int)

        ploty = ploty.astype(int)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), lane_color)

        return color_warp