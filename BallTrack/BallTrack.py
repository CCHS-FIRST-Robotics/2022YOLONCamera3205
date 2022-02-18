import numpy as np
import cv2
from BallTrack.BallDetect.BallColorDetect import BallColorDetect
from BallTrack.BallDetect.BallPCDetect import BallPCDetect


class BallTrack:
    def __init__(self, R_COL, B_COL, AREA_PROP, COL_PROP, BALL_RADIUS, X_FOV, CHECK_RING):
        self.bcd = BallColorDetect(R_COL, B_COL, AREA_PROP, COL_PROP)
        self.bpd = BallPCDetect(BALL_RADIUS, X_FOV, CHECK_RING)

    def getBalls(self, pc, snp, disp):
        post_points, r_mask, b_mask = self.bcd.ballDetect(snp, disp)
        points = self.bpd.verifyAll(pc, post_points)
        return points, r_mask, b_mask

    def updateTrack(self, odo, pc, snp):
        points = self.getBalls(pc, snp)
        for point in points:
            xyz = pc[point[0:2], :]
            xy = xyz[0:2]
            xy.shape = (1, -1)
            xy = odo.npTransform(xy)
            xyz[0:2] = xy
           