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
        new_points = points[inrange, :]
        return new_points

    def getDists(self, pc, ring):
        if ring.shape[0] == 0:
            return np.zeros([0])
        ring = ring.astype(int)
        points = pc[ring[:, 0], ring[:, 1], 0] ** 2 + pc[ring[:, 0], ring[:, 1], 1] ** 2 + pc[ring[:, 0], ring[:, 1], 2] ** 2
        points = points ** 0.5
        return points

    def verifyBall(self, pc, coord):
        x = coord[0]
        y = coord[1]
        coord[0] = y
        coord[1] = x
        dist = self.np2mag(pc[round(coord[0]), round(coord[1]), :])
        it_dist = self.BALL_RADIUS * self.CHECK_RING[0]
        ot_dist = self.BALL_RADIUS * self.CHECK_RING[1]
        x_size = pc.shape[1]
        pixel_p_rad = x_size / (self.X_FOV)

        ball_pixel = self.dist2pix(dist, self.BALL_RADIUS, pixel_p_rad)
        pixel_err = abs(ball_pixel - coord[3])
        print("Ball pixel", pixel_err, dist, ball_pixel, coord[3])

        #itp = self.dist2pix(dist, it_dist, pixel_p_rad)
        #otp = self.dist2pix(dist, ot_dist, pixel_p_rad)
        itp = coord[3] * self.CHECK_RING[0]
        otp = coord[3] * self.CHECK_RING[1]
        it_ring = self.checkRing(pc, coord, itp)
        ot_ring = self.checkRing(pc, coord, otp)
        it_d = self.getDists(pc, it_ring)
        ot_d = self.getDists(pc, ot_ring)
        if it_d.shape[0] == 0:
            its = 0.01
        else:
            its = np.mean(np.abs(it_d - dist) < self.BALL_RADIUS * 0.7)
        if ot_d.shape[0] == 0:
            ots = 0.01
        else:
            ots = np.mean(np.abs(ot_d - dist) > self.BALL_RADIUS * 0.7)
        return ( its*2 + ots > 2.3 )and (pixel_err < 50)

    def verifyAll(self, pc, coords):
        balls = []
        print(coords)
        for coord in coords:
            try:
                verify = self.verifyBall(pc, coord.copy())
            except:
                verify = False
            if verify:
                balls += [coord]
        return balls