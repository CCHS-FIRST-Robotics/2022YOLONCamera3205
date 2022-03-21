import cv2
import numpy as np

BALL_RADIUS = 0.12
CAM_DIST = 0.195
RPP = 2 * 3.1415 * 53 / (360 * 640)

class Ball2LPos:
    def __init__(self, img_shape):
        self.img_shape = img_shape
        print("Ball2LPos")
        l_mat = np.genfromtxt("new_mtxL.csv", delimiter=',')
        r_mat = np.genfromtxt("new_mtxR.csv", delimiter=',')
        distL = np.genfromtxt("distL.csv", delimiter=',')
        distR = np.genfromtxt("distR.csv", delimiter=',')

        self.cmat = l_mat

        self.rev_proj_matrix = np.zeros((4, 4))  # to store the output
        cv2.stereoRectify(cameraMatrix1=l_mat, cameraMatrix2=r_mat,
                          distCoeffs1=distL, distCoeffs2=distR,
                          imageSize=self.img_shape,
                          R=np.identity(3), T=np.array([0.195, 0., 0.]),
                          R1=None, R2=None,
                          P1=None, P2=None,
                          Q=self.rev_proj_matrix)
    def radius2JankDisp(self, lball):
        ball_angle = RPP * lball[3]
        # tan(theta) = radius / dist
        distance = BALL_RADIUS / np.tan(ball_angle)
        # pixel focal lenght / distance
        return self.cmat[0,0] * CAM_DIST / distance

    def getDisparity(self, lball, rballs):
        color = lball[2]
        min_rad = 50000
        min_index = -1
        for c in range(len(rballs)):
            ball = rballs[c]
            if ball[2] == color and ball[0] >= lball[0]:
                rad_diff = abs(ball[3] - lball[3])
                min_y_disp = abs(rballs[min_index][1] - lball[1])
                if rad_diff < min_rad and min_y_disp < 40:
                    min_rad = rad_diff + min_y_disp
                    min_index = c
        if min_index != -1:
            disp = abs(lball[0] - rballs[min_index][0])
            rballs = rballs.pop(min_index)
            return disp
        return 0

    def makeBallList(self, lballs, rballs):
        ball_list = []
        temp_img = np.zeros(self.img_shape[0:2])
        rballs2 = rballs.copy()
        for ball in lballs:
            disp = self.radius2JankDisp(ball)
            sdisp = self.getDisparity(ball, rballs2)
            if sdisp != 0:
                disp = disp * 0.3 + sdisp * 0.7
            if abs(sdisp - disp) < 40:
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