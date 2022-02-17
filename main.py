import numpy as np
import math
from CameraHandler import CameraHandler
from DeepPruner.NetDeploy import DPNN
from Odometry import Odometry
from display import Display
from BallTrack.BallTrack import BallTrack
from PathFinder.ObstacleDetect import ObstacleDetect
import cv2
import time

# ind 0 is the width, ind 1 is the height
CAMERA_SIZE = (1280, 720)
CROP_BORDER_SIZE = (32, 48)
DOWNSCALE_FACTOR = (1, 1)
# ind 0 is Left, ind 2 is Right
CAM_PORTS = (1, 2)

CAM_DIST = 0.15

X_FOV = (2 * math.pi) * 53 / 360

DISP_CAL = np.array([[1., 0., 0., -292.3226819],
                     [0., 1., 0., -228.78747177],
                     [0., 0., 0., 499.35464466],
                     [0., 0., -6.80272109, 0.]])

PRUNED_PC_SIZE = 2500

AREA_PROP = 0.4
COL_PROP = 0.15

R_COL = [0, 62, 80, 20, 220, 255]
B_COL = [89, 30, 60, 125, 220, 220]

BALL_RADIUS = 0.12

LOCAL_POS = [-0.2, 0.3, 0.5]

OBS_H_RANGE = [BALL_RADIUS + 0.03, 0.6]
GRID_SIZE = 0.1
TOWER_RAD = 1.5
ROBOT_RAD = 0.4

class main:
    def __init__(self):
        self.cam_hand = CameraHandler(CAMERA_SIZE, CROP_BORDER_SIZE, DOWNSCALE_FACTOR, CAM_PORTS)
        self.dp_pc = DPNN(CAM_DIST, DISP_CAL, PRUNED_PC_SIZE)
        self.ball_track = BallTrack(R_COL,B_COL,AREA_PROP, COL_PROP, BALL_RADIUS, X_FOV, [0.5, 1.5])
        self.odo = Odometry(LOCAL_POS)
        self.obsd = ObstacleDetect(OBS_H_RANGE, GRID_SIZE, TOWER_RAD, ROBOT_RAD)
        self.display = Display()

    def update(self):
        start_time = time.time()
        l, r = self.cam_hand.snapshot()
        disp, pruned_pc, pc = self.dp_pc.makePC(l, r)
        balls = self.ball_track.getBalls(pc, l)
        map = self.obsd.updateMap(self.odo, pruned_pc)
        self.display.display(l, balls, disp)
        print("Elapsed Time: {}".format(time.time() - start_time))

m = main()
if __name__ == "__main__":
    while True:
        m.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()