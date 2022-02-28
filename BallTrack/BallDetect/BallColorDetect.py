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
        #snp_g = cv2.medianBlur(snp_g, 1)
        rows = snp_g.shape[0]
        disp[disp == None] = 0
        disp = disp/np.max(disp)
        disp = disp * 255
        disp = (disp + snp_g) * 0.5
        disp = disp.astype(np.uint8)
        #disp = cv2.medianBlur(disp, 3)
        circles = cv2.HoughCircles(disp, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=100, param2=50,
                                   minRadius=20, maxRadius=60)
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

    def colorContour(self, snp, col_range, tag, col_thresh):
        color_l = np.array(col_range[0:3])
        color_h = np.array(col_range[3:6])
        snp = cv2.GaussianBlur(snp, (31,31),0)
        #snp = cv2.bilateralFilter(snp, 5, 75, 75)
        snp_hsv = cv2.cvtColor(snp, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(snp_hsv, color_l, color_h)
        #kernel = np.ones((3, 3), np.uint8)
        #mask = cv2.erode(mask, np.ones((3,3), np.uint8) ,iterations = 5)
        #mask = cv2.dilate(mask, np.ones((3,3), np.uint8) ,iterations = 3)
        #mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations = 2)
        #mask = cv2.dilate(mask, np.ones((7,7), np.uint8) ,iterations = 4)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        postulate_ball_coords = []
        
        target_hsv = (color_l[0] + color_l[1])*0.5

        for i, contour in enumerate(contours):
            ((x0, y0), radius) = cv2.minEnclosingCircle(contour)
            circle_m = np.zeros(mask.shape, np.uint8)
            cv2.circle(circle_m, (round(x0), round(y0)), round(radius), (255,255,255), -1)
            upper = np.ones([round(y0), snp.shape[1]])
            lower = np.zeros([snp.shape[0] - round(y0), snp.shape[1]])
            ud_mask = np.concatenate([upper, lower],axis = 0)

            area = np.sum(mask * ud_mask * circle_m)
            
            #gen_area = (circle_m * snp_hsv[:,:,0])
            #area_mask = gen_area != 0
            #gen_area = np.abs(gen_area - target_hsv) 
            #gen_area = np.exp(-1 * ((gen_area - target_hsv)/20)**2) * 2
            #gen_area = np.sum(gen_area[area_mask])
            
            #area = cv2.contourArea(contour)
            mean_color = cv2.mean(snp_hsv, mask = circle_m)
            
            color_diff_mag = np.abs(mean_color[0] - target_hsv )
            print(area, (np.pi * radius ** 2), color_diff_mag)


            if radius > 2 and area * 2 > self.AREA_PROP * (np.pi * radius ** 2) and color_diff_mag < col_thresh:
                postulate_ball_coords += [[round(x0), round(y0), tag, radius]]

        return postulate_ball_coords, mask

    def ballDetect(self, snp, disp):
        print("R Cols")
        red, rmask = self.colorContour(snp, self.R_COL, "R", 90)
        print("B Cols")
        blue, bmask = self.colorContour(snp, self.B_COL, "B", 40)
        #hough = self.houghDetect(snp, disp)
        ball_list = red + blue# + hough
        #ball_list = hough
        return ball_list, rmask, bmask