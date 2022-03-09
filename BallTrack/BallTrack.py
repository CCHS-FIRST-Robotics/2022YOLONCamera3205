import numpy as np
import cv2
from BallTrack.BallDetect.BallColorDetect import BallColorDetect
from BallTrack.BallDetect.BallPCDetect import BallPCDetect
from BallTrack.Ball import Ball
import time


class BallTrack:
    def __init__(self, R_COL, B_COL, AREA_PROP, COL_PROP, BALL_RADIUS, X_FOV, CHECK_RING, N=40):
        self.ball_list = [Ball(BALL_RADIUS)] * N
        self.prev_time = time.time()

    def updateTrack(self, lball, rball, p2p, img_shape, odo):
        options_list = []
        dt = time.time() - self.prev_time
        self.prev_time = time.time()
        for i in range(len(lball)):
            options_list += [i]
        for ball in self.ball_list:
            if ball.state != 0:
                ball.predict(dt)
                min_dist = 999
                option = -1
                o_pt = [0, 0, 0]
                for c in range(len(options_list)):
                    try:
                        point = lball[options_list[c]]
                        xyz = np.array(p2p(img_shape, lball, rball, options_list[c]))
                        xyz.shape = (1, -1)
                        xyz = odo.npTransform(xyz)
                        if point[2] == "R" and ball.color == 0:
                            dist = ball.getDist(xyz)
                            if dist < min_dist:
                                min_dist = dist
                                option = c
                                o_pt = xyz
                    except:
                        print("oob")
                if min_dist < ball.getSpd() * dt + 0.2:
                    ball.update(o_pt)
                    options_list.pop(option)
                elif ball.fresh == 0:
                    ball.reset()
                if time.time() - ball.last_tracked > 1.5:
                    ball.reset()
        n = 0
        while len(options_list) > 0 and n < len(self.ball_list):
            if self.ball_list[n].state == 0:
                point = lball[options_list[0]]
                xyz = np.array(p2p(img_shape, lball, rball, options_list[0]))
                xyz.shape = (1, 3)
                xyz = odo.npTransform(xyz)
                xyz.shape = (-1)
                print(xyz)
                self.ball_list[n].init(list(xyz), point[2])
                if (len(options_list) > 0):
                   options_list.pop(0)
            n += 1
        return self.ball_list