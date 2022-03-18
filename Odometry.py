import numpy as np


class Odometry:
    def __init__(self, LOCAL_POS):
        self.r_pos = np.array([0, 0])
        self.l_pos = np.array(LOCAL_POS)
        self.l_pos.shape = [1, -1]
        self.heading = 0

    def setRobotPos(self, pos, heading):
        self.r_pos = pos
        self.heading = heading

    def npTransform(self, coords):
        coords = coords + np.tile(self.l_pos, [coords.shape[0], 1])
        xy = coords[:,0:2]
        xy.shape = [2, -1]
        rot_mat = np.array(
            [[np.cos(self.heading), -1 * np.sin(self.heading)], [np.sin(self.heading), np.cos(self.heading)]])
        rotated_xy = np.matmul(rot_mat, xy)
        rotated_xy.shape = [-1, 2]
        pos_t = self.r_pos.copy()
        pos_t.shape = (1, -1)
        #rotated_xy2 = np.zeros([rotated_xy.shape[0], 3])
        #rotated_xy2[:, 0:2] = rotated_xy
        new_xy = rotated_xy + np.tile(pos_t[:,0:2], [rotated_xy.shape[0], 1])
        coords[:,0:2] = new_xy
        return coords