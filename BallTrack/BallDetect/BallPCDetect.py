import numpy as np


class BallPCDetect:
    def __init__(self, BALL_RADIUS, X_FOV, CHECK_RING):
        self.BALL_RADIUS = BALL_RADIUS
        self.X_FOV = X_FOV
        self.CHECK_RING = CHECK_RING

    def np2mag(self, point):
        return (point[0] ** 2 + point[1] ** 2 + point[2] ** 2) ** 0.5

    def dist2pix(self, dist, tan_dist, pixel_p_radian):
        return pixel_p_radian * np.arctan(tan_dist / dist)

    def checkRing(self, pc, coord, radius):
        r2o2 = (2 ** 0.5) / 2
        points = []
        points += [[coord[0], coord[1] + radius]]
        points += [[coord[0] + radius * r2o2, coord[1] + radius * r2o2]]
        points += [[coord[0] + radius, coord[1]]]
        points += [[coord[0] + radius * r2o2, coord[1] - radius * r2o2]]
        points += [[coord[0], coord[1] - radius]]
        points += [[coord[0] - radius * r2o2, coord[1] - radius * r2o2]]
        points += [[coord[0] - radius, coord[1]]]
        points += [[coord[0] - radius * r2o2, coord[1] + radius * r2o2]]
        points = np.array(points)
        points = np.round(points)
        x_inrange = np.logical_and(points[:, 0] >= 0, points[:, 0] < pc.shape[0])
        y_inrange = np.logical_and(points[:, 1] >= 1, points[:, 1] < pc.shape[1])
        inrange = np.logical_and(x_inrange, y_inrange)
        new_points = points[inrange,:]

    def verifyBall(self, pc, coord):
        dist = self.np2mag(pc[coord[0], coord[1], :])
        it_dist = self.BALL_RADIUS * self.CHECK_RING[0]
        ot_dist = self.
