import numpy as np
import cv2
from BallTrack.BallDetect.BallColorDetect import BallColorDetect
from BallTrack.BallDetect.BallPCDetect import BallPCDetect
from BallTrack.Ball import Ball
import time


class BallTrack:
    def __init__(self, R_COL, B_COL, AREA_PROP, COL_PROP, BALL_RADIUS, X_FOV, CHECK_RING, N=40):
        self.bcd = BallColorDetect(R_COL, B_COL, AREA_PROP, COL_PROP)
        self.bpd = BallPCDetect(BALL_RADIUS, X_FOV, CHECK_RING)
        self.ball_list = [Ball(BALL_RADIUS)] * N
        self.prev_time = time.time()

    def getBalls(self, pc, snp, disp):
        post_points, r_mask, b_mask = self.bcd.ballDetect(snp, disp)
        points = self.bpd.verifyAll(pc, post_points)
        return points, r_mask, b_mask

    def updateTrack(self, odo, pc, snp):
        points, r_mask, b_mask = self.getBalls(pc, snp)
        options_list = []
        dt = time.time() - self.prev_time
        self.prev_time = time.time()
        for i in range(len(points)):
            options_list += [i]
        for ball in self.ball_list:
            if ball.state != 0:
                ball.predict(dt)
                min_dist = 999
                option = -1
                o_pt = [0, 0, 0]
                for c in range(len(options_list)):
                    point = points[options_list[c]]
                    xyz = pc[point[0:2], :]
                    xyz.shape = (1, -1)
                    xyz = odo.npTransform(xyz)
                    if point[2] == "R" and ball.color == 0:
                        dist = ball.getDist(xyz)
                        if dist < min_dist:
                            min_dist = dist
                            option = c
                            o_pt = xyz
                if min_dist < ball.getSpd() * dt + 0.2:
                    ball.update(o_pt)
                    options_list.pop(option)
                if time.time() - ball.last_tracked > 1.5:
                    ball.reset()
        n = 0
        while len(options_list) > 0 and n < len(self.ball_list):
            if self.ball_list[c].state == 0:
                point = points[options_list[0]]
                xyz = pc[point[0:2], :]
                xyz.shape = (1, -1)
                xyz = odo.npTransform(xyz)
                self.ball_list[c].init(xyz, points[2])
                options_list.pop(0)
            n += 1
        return points, r_mask, b_mask