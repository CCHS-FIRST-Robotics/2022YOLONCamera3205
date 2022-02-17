import numpy as np
import cv2
from BallTrack.BallDetect.BallColorDetect import BallColorDetect
from BallTrack.BallDetect.BallPCDetect import BallPCDetect


class BallTrack:
    def __init__(self, R_COL, B_COL, AREA_PROP, COL_PROP, BALL_RADIUS, X_FOV, CHECK_RING):
        self.bcd = BallColorDetect(R_COL, B_COL, AREA_PROP, COL_PROP)
        self.bpd = BallPCDetect(BALL_RADIUS, X_FOV, CHECK_RING)

    def getBalls(self, pc, snp):
        post_points = self.bcd.ballDetect(snp)
        points = self.bpd.verifyAll(pc, post_points)
        return points