import numpy as np
import cv2


class BallColorDetect:
    def __init__(self, R_COL, B_COL, AREA_PROP, COL_PROP):
        self.R_COL = R_COL
        self.B_COL = B_COL
        self.AREA_PROP = AREA_PROP
        self.COL_PROP = COL_PROP
        self.R_HUE = (R_COL[0] + R_COL[3]) * 0.5
        self.B_HUE = (B_COL[0] + B_COL[3]) * 0.5

    def houghDetect(self, snp):
        snp_hsv = cv2.cvtColor(snp, cv2.COLOR_BGR2HSV)
        snp_g = cv2.cvtColor(snp, cv2.COLOR_BGR2GRAY)
        rows = snp_g.shape[0]
        circles = cv2.HoughCircles(snp_g, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=100, param2=30,
                                   minRadius=4, maxRadius=50)
        postulate_ball_coords = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                try:
                    hue = snp_hsv[i[0], i[1], 0]
                    tag = "R"
                    if abs(hue - self.B_HUE) < abs(hue - self.R_HUE):
                        tag = "B"
                    center = [i[0], i[1], tag]
                    postulate_ball_coords += [center]
                except:
                    print("oor")
        
        return postulate_ball_coords

    def colorContour(self, snp, col_range, tag):
        color_l = np.array(col_range[0:3])
        color_h = np.array(col_range[3:6])
        snp = cv2.bilateralFilter(snp, 5, 75, 75)
        snp_hsv = cv2.cvtColor(np.float32(snp), cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(np.float32(snp_hsv), color_l, color_h)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=5)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
                postulate_ball_coords += [[x0, y0, tag]]

        return postulate_ball_coords

    def ballDetect(self, snp):
        red = self.colorContour(snp, self.R_COL, "R")
        blue = self.colorContour(snp, self.B_COL, "B")
        hough = self.houghDetect(snp)
        ball_list = red + blue + hough
        return ball_list