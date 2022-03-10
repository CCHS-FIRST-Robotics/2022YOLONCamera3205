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
    def getDisparity(self, given_ball, rballs):
        color = given_ball[2]
        min_y_disp = 50000
        min_index = -1
        for c in range(len(rballs)):
            ball = rballs[c]
            if ball[2] == color:
                print("match", ball[0], given_ball[0])
                if ball[0] >= given_ball[0]:
                    if abs(ball[1] - given_ball[1]) < min_y_disp:
                        min_y_disp = abs(ball[1] - given_ball[1])
                        min_index = c
        if min_y_disp < 10:
            return abs(given_ball[0] - rballs[min_index][0]) + 1
        return 0
    def makeBallList(self, lballs, rballs):
        ball_list = []
        temp_img = np.zeros(self.img_shape[0:2])
        for ball in lballs:
            disp = self.getDisparity(ball, rballs)
            if disp != 0:
                ball_list += [ball]
                temp_img[ball[1], ball[0]] = disp
        temp_img = temp_img.astype(np.uint8)
        points = cv2.reprojectImageTo3D(temp_img, self.rev_proj_matrix)
        n_ball_list = []
        for c in range(len(ball_list)):
            val = points[ball_list[c][1], ball_list[c][0], :]
            if np.sum(np.isnan(val)) == 0:
                n_ball_list += [ball_list[c] + list(val)]
        return n_ball_list