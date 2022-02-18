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

    def houghDetect(self, snp, disp):
        snp_hsv = cv2.cvtColor(snp, cv2.COLOR_BGR2HSV)
        snp_g = cv2.cvtColor(snp, cv2.COLOR_BGR2GRAY)
        snp_g = cv2.medianBlur(snp_g, 9)
        rows = snp_g.shape[0]
        disp[disp == None] = 0
        disp = disp/np.max(disp)
        disp = disp.astype(float)
        circles = cv2.HoughCircles(disp, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=30, param2=15,
                                   minRadius=20, maxRadius=100)
        postulate_ball_coords = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            #print(circles)
            for i in circles[0, :]:
                try:
                    hue = snp_hsv[i[0], i[1], 0]
                    tag = "R"
                    if abs(hue - self.B_HUE) < abs(hue - self.R_HUE):
                        tag = "B"
                    center = [round(i[0]), round(i[1]), tag, i[2]]
                    postulate_ball_coords += [center]
                except:
                    print("oor")
        
        return postulate_ball_coords

    def colorContour(self, snp, col_range, tag):
        color_l = np.array(col_range[0:3])
        color_h = np.array(col_range[3:6])
        snp = cv2.bilateralFilter(snp, 5, 75, 75)
        snp_hsv = cv2.cvtColor(snp, cv2.COLOR_BGR2HSV)
        
        print(np.max(snp_hsv))
        mask = cv2.inRange(snp_hsv, color_l, color_h)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, np.ones((3,3), np.uint8) ,iterations = 2)
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8) ,iterations = 10)
        mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations = 2)
        mask = cv2.dilate(mask, np.ones((7,7), np.uint8) ,iterations = 4)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        postulate_ball_coords = []

        for i, contour in enumerate(contours):
            ((x0, y0), radius) = cv2.minEnclosingCircle(contour)

            # color test thing #
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)  # offsets - with this you get 'mask'
            min_rect = cv2.minAreaRect(contour)
            color = np.array(cv2.mean(mask[y:y + h, x:x + w])).astype(np.uint8)[0]

            rad_check = radius > 4
            area_check = area > self.AREA_PROP * (np.pi * radius ** 2)

            if radius > 4 and cv2.contourArea(contour) > self.AREA_PROP * (np.pi * radius ** 2) and (
                    color / 255) > self.COL_PROP:
                postulate_ball_coords += [[round(x0), round(y0), tag, radius]]

        return postulate_ball_coords, mask

    def ballDetect(self, snp, disp):
        red, rmask = self.colorContour(snp, self.R_COL, "R")
        blue, bmask = self.colorContour(snp, self.B_COL, "B")
        hough = self.houghDetect(snp, disp)
        #ball_list = red + blue + hough
        ball_list = hough
        return ball_list, rmask, bmask