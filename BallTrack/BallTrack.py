import numpy as np
import cv2
from BallTrack.BallDetect.BallColorDetect import BallColorDetect
from BallTrack.BallDetect.BallPCDetect import BallPCDetect
from BallTrack.Ball import Ball
import time


class BallTrack:
    def __init__(self, R_COL, B_COL, AREA_PROP, COL_PROP, BALL_RADIUS, X_FOV, CHECK_RING, N=40):
        self.ball_list = []
        for c in range(N):
            self.ball_list += [Ball(BALL_RADIUS)]
        self.prev_time = time.time()

    def updateTrack(self, nball_list, img_shape, odo):
        options_list = []
        dt = time.time() - self.prev_time
        self.prev_time = time.time()
        for i in range(len(nball_list)):
            options_list += [i]
        a = 0
        for ball in self.ball_list:
            if ball.state != 0:
                a += 1
                #ball.predict(dt)
                min_dist = 999
                option = -1
                o_pt = np.array([[0, 0, 0]])
                for c in range(len(options_list)):
                    point = nball_list[options_list[c]]
                    xyz = np.array(nball_list[options_list[c]][4:7])
                    xyz.shape = (1, -1)
                    xyz = odo.npTransform(xyz)
                    if (point[2] == "R" and ball.color == 0) or (point[2] == "B" and ball.color == 1):
                        dist = ball.getDist(xyz[0,:])
                        if dist < min_dist:
                            min_dist = dist
                            option = c
                            o_pt = xyz
                if min_dist < ball.getSpd() * dt + 0.5:
                    ball.update(o_pt[0,:])
                    
                    if (len(options_list) > 0):
                        options_list.pop(option)
                elif ball.fresh == 0:
                    ball.reset()
                if time.time() - ball.last_tracked > 1.5:
                    ball.reset()
        n = 0
        debug = []
        while len(options_list) > 0 and n < len(self.ball_list):
            if self.ball_list[n].state == 0:
                point = nball_list[options_list[0]]
                xyz = np.array(nball_list[options_list[0]][4:7])
                xyz.shape = (1, 3)
                xyz = odo.npTransform(xyz)
                xyz.shape = (-1)
                debug += [n]
                ball = self.ball_list[n]
                ball.init(list(xyz), point[2], state = 2)
                if (len(options_list) > 0):
                   options_list.pop(0)
            n += 1
        return self.ball_list