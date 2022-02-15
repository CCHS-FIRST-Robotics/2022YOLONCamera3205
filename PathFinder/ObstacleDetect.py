import numpy as np
import open3d as o3d
import math
import time

class ObstacleDetect:
    def __init__(self):
        self.pos = [0,0]
        self.heading = 0
        self.height = 0.5
        self.obs_h_range = [0.3, 0.6]
        field_size = (8.23, 16.46)
        self.grid_size = 0.1
        y_size = int(field_size[0]/self.grid_size)
        y_size = y_size + y_size%2 + 1
        x_size = int(field_size[1]/self.grid_size)
        x_size = x_size + x_size%2 + 1
        self.map = np.zeros([y_size, x_size])

    def makeInitTrans(self):
        trans_init = np.asarray([[np.cos(self.heading), -1 * np.sin(self.heading), 0, self.pos[0]],
                                 [np.sin(self.heading), np.cos(self.heading), 0, self.pos[1]], [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        return trans_init

    def o3dProcess(self, pc):
        pc[2] = pc[2] + self.height
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        #pcd.voxel_down_sample(0.4)
        trans_init = self.makeInitTrans()
        pcd.transform(trans_init)
        
        return pcd

    def point2Cell(self, coords):
        if coords[2] > self.obs_h_range[0] and coords[2]< self.obs_h_range[1]:
            return False, (0,0), 0
        #0.15/0.1 = 1.5
        if (np.sum(np.isnan(coords))) > 0:
            return False, (0,0), 0
        weight = np.exp(-1 * (coords[2] - 0.3)**2 / (16 * self.height ** 2))
        x_cell = int(coords[0] / self.grid_size) + int(self.map.shape[1]/2)
        y_cell = int(self.map.shape[0]/2) - int(coords[1] / self.grid_size)
        if (x_cell < 0 or x_cell > self.map.shape[1] - 1):
            return False, (0, 0), 0
        if (y_cell < 0 or y_cell > self.map.shape[0] - 1):
            return False, (0, 0), 0
        return True, (y_cell, x_cell), 1


    def updateMap(self, pc):
        pcd = self.o3dProcess(pc)
        point_np = np.asarray(pcd.points)
        height_mask = np.logical_and(point_np[:,2] > self.obs_h_range[0] , point_np[:,2] < self.obs_h_range[1])
        filtered_points = point_np[height_mask, :]
        x_coord = np.floor(filtered_points[:,0] / self.grid_size) + int(self.map.shape[1]/2)
        y_coord = np.floor(filtered_points[:,1] / self.grid_size) + int(self.map.shape[0]/2)
        coords = np.array([y_coord,x_coord])
        size_mask =  np.logical_and(np.logical_and(coords[:,0] > 0, coords[:,1] > 0), np.logical_and(coords[:,0] < self.map.shape[0], coords[:,1] < self.map.shape[1]))
        coords = coords[size_mask, :]

        temp_map = np.zeros(self.map.shape)
        for c1 in range(coords.shape[0]):
            temp_map[int(coords[c1,0]), int(coords[c1,1])] += 1
        return temp_map

    def updateMapL(self, pc):
        st = time.time()
        pcd = self.o3dProcess(pc)
        point_np = np.asarray(pcd.points)
        print(np.mean(point_np[:,1]))
        self.map = self.map * 0.5
        #height_mask = np.logical_and(point_np[:,2] > self.obs_h_range[0] , point_np[:,2] < self.obs_h_range[1])
        #filtered = point_np[height_mask,:]
        filtered = point_np
        for c in range(filtered.shape[0]):
            a, b, w = self.point2Cell(filtered[c,:])
            if a:
                self.map[b[0],b[1]] += 0.5 * w
        return self.map