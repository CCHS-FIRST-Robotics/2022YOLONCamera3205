import numpy as np


class Odometry:
    def __init__(self, LOCAL_POS):
        self.r_pos = [0, 0]
        self.l_pos = LOCAL_POS
        self.heading = 0

    def setRobotPos(self, pos, heading):
        self.r_pos = pos
        self.heading = 0

    def getBallPos(self, pc, coord):
        loc = pc[coord[0], coord[1], :]
        xy = loc[0:2]
        xy.shape = [1, 2]
        rot_mat = np.array(
            [[np.cos(self.heading), -1 * np.sin(self.heading)], [np.sin(self.heading), np.cos(self.heading)]])
