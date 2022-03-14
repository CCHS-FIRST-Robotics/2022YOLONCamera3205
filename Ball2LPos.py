import cv2
import numpy as np


class Ball2LPos:
    def __init__(self, img_shape):
        self.img_shape = img_shape
        print("Ball2LPos")
        l_mat = np.genfromtxt("new_mtxL.csv", delimiter=',')
        r_mat = np.genfromtxt("new_mtxR.csv", delimiter=',')
        distL = np.genfromtxt("distL.csv", delimiter=',')
        distR = np.genfromtxt("distR.csv", delimiter=',')

        self.rev_proj_matrix = np.zeros((4, 4))  # to store the output
        cv2.stereoRectify(cameraMatrix1=l_mat, cameraMatrix2=r_mat,
                          distCoeffs1=distL, distCoeffs2=distR,
                          imageSize=self.img_shape,
                          R=np.identity(3), T=np.array([0.195, 0., 0.]),
                          R1=None, R2=None,
                          P1=None, P2=None,
                          Q=self.rev_proj_matrix)
    def getDisparityOld(self, given_ball, rballs):
        color = given_ball[2]
        min_y_disp = 50000
        min_index = -1
        for c in range(len(rballs)):
            ball = rballs[c]
            if ball[2] == color:
                print("match", ball[0], given_ball[0])
                if ball[0] <= given_ball[0]:
                    if abs(ball[1] - given_ball[1]) < min_y_disp:
                        min_y_disp = abs(ball[1] - given_ball[1])
                        min_index = c
        if min_y_disp < 40:
            return abs(given_ball[0] - rballs[min_index][0])
        
        return 0

    def getDisparity(self, lball, rballs):
        color = lball[2]
        min_rad = 50000
        min_index = -1
        for c in range(len(rballs)):
            ball = rballs[c]
            if ball[2] == color and ball[0] <= lball[0]:
                rad_diff = abs(ball[3] - lball[3])
                if rad_diff < min_rad:
                    min_rad = rad_diff
                    min_index = c
        if min_index != -1:
            min_y_disp = abs(rballs[min_index][1] - lball[1])
            if min_y_disp < 40:
                return abs(lball[0] - rballs[min_index][0])
        return 0

    def makeBallList(self, lballs, rballs):
        ball_list = []
        temp_img = np.zeros(self.img_shape[0:2])
        for ball in lballs:
            disp = self.getDisparity(ball, rballs)
            if disp != 0:
                print("disp", disp)
                ball_list += [ball]
                temp_img[ball[1], ball[0]] = disp
        temp_img = temp_img.astype(np.uint8)
        points = cv2.reprojectImageTo3D(temp_img, self.rev_proj_matrix)
        # 3 is negative y
        # 1 is negative x
        # 2 is z correct
        n_ball_list = []
        for c in range(len(ball_list)):
            val = points[ball_list[c][1], ball_list[c][0], :]
            new_val = [val[0] * -1, val[2] * -1, val[1]]
            if np.sum(np.isnan(val)) == 0:
                n_ball_list += [ball_list[c] + new_val]
        return n_ball_list