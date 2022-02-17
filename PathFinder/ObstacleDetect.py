import numpy as np
import math
import time
import cv2


class ObstacleDetect:
    def __init__(self, OBS_H_RANGE, GRID_SIZE, TOWER_RAD, ROBOT_RAD, THRESH=10):
        self.obs_h_range = OBS_H_RANGE
        field_size = (8.23, 16.46)
        self.grid_size = GRID_SIZE
        self.tower_rad = TOWER_RAD
        self.THRESH = THRESH
        y_size = int(field_size[0] / self.grid_size)
        y_size = y_size + y_size % 2 + 1
        x_size = int(field_size[1] / self.grid_size)
        x_size = x_size + x_size % 2 + 1
        self.map = np.zeros([y_size, x_size])
        #blot out square of robot location + rad
        self.r_size_shift = int(ROBOT_RAD / self.grid_size) + 1
        self.makeStaticMap()

    def makeStaticMap(self):
        self.static_map = np.zeros(self.map.shape)
        center = (int(self.map.shape[0] / 2), int(self.map.shape[1] / 2))
        for y in range(self.static_map.shape[0]):
            for x in range(self.static_map.shape[1]):
                if x == 0:
                    self.static_map[y, x] = 1
                if x == self.static_map.shape[1] - 1:
                    self.static_map[y, x] = 1
                if y == 0:
                    self.static_map[y, x] = 1
                if y == self.static_map.shape[0] - 1:
                    self.static_map[y, x] = 1
                dist = ((center[0] - y) ** 2 + (center[1] - x) ** 2) ** 0.5
                dist = dist * self.grid_size
                if dist < self.tower_rad:
                    self.static_map[y, x] = 1

    def point2Cell(self, coords):
        if coords[2] > self.obs_h_range[0] and coords[2] < self.obs_h_range[1]:
            return False, (0, 0), 0
        if (np.sum(np.isnan(coords))) > 0:
            return False, (0, 0), 0
        x_cell = int(coords[0] / self.grid_size) + int(self.map.shape[1] / 2)
        y_cell = int(self.map.shape[0] / 2) - int(coords[1] / self.grid_size)
        if (x_cell < 0 or x_cell > self.map.shape[1] - 1):
            return False, (0, 0), 0
        if (y_cell < 0 or y_cell > self.map.shape[0] - 1):
            return False, (0, 0), 0
        return True, (y_cell, x_cell), 1

    def xyr2Cells(self, coord):
        if (np.sum(np.isnan(coord))) > 0:
            return self.map
        x_cell = int(coord[0] / self.grid_size) + int(self.map.shape[1] / 2)
        y_cell = int(self.map.shape[0] / 2) - int(coord[1] / self.grid_size)
        for y in range(self.r_size_shift*2 + 1):
            for x in range(self.r_size_shift*2 + 1):
                new_coord = (y_cell + y - self.r_size_shift, x_cell + x - self.r_size_shift)
                try:
                    self.map[new_coord[0], new_coord[1]] = 1
                except:
                    print("rloc oor")
        return self.map

    def updateMap(self, odo, pc):
        point_np = odo.npTransform(pc)
        self.map = self.map * 0.5
        for c in range(point_np.shape[0]):
            a, b, w = self.point2Cell(point_np[c, :])
            if a:
                self.map[b[0], b[1]] += 0.5 * w
        #remove current_loc
        self.xyr2Cells(odo.r_pos)
        final_map = self.static_map * self.THRESH * 2 + self.map
        final_map = final_map > self.THRESH
        final_map = final_map.astype(float)
        final_map = cv2.dilate(final_map, np.ones([3,3]), iterations = 1)
        return final_map
