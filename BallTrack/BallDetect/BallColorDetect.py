import numpy as np
import cv2


class BallColorDetect:
    def __init__(self, R_COL, B_COL, AREA_PROP, COL_PROP):
        self.R_COL = R_COL
        self.B_COL = B_COL
        self.AREA_PROP = AREA_PROP
        self.COL_PROP = COL_PROP

    def colorContour(self, snp, col_range):
        color_l = np.array(col_range[0:3])
        color_h = np.array(col_range[2:6])
        snp_hsv = cv2.cvtColor(snp, cv2.COLOR_BGR2HSV)
        snp_g = cv2.cvtColor(snp, cv2.COLOR_BGR2GRAY)

        mask = cv2.inRange(snp_hsv, color_l, color_h)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        canny = cv2.Canny(snp_g, 100, 200)

        postulate_ball_coords = []

        for i, contour in enumerate(contours):
            ((x0, y0), radius) = cv2.minEnclosingCircle(contour)

            # color test thing #
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)  # offsets - with this you get 'mask'
            min_rect = cv2.minAreaRect(contour)
            color = np.array(cv2.mean(mask[y:y + h, x:x + w])).astype(np.uint8)[0]

            rad_check = radius > 8
            area_check = area > self.AREA_PROP * (np.pi * radius ** 2)

            if radius > 8 and cv2.contourArea(contour) > self.AREA_PROP * (np.pi * radius ** 2) and (
                    color / 255) > self.COL_PROP:
                postulate_ball_coords += [(x0, y0)]

        return postulate_ball_coords