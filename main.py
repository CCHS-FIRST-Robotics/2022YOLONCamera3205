import numpy as np
import math
from CameraHandler import CameraHandler
from DeepPruner.NetDeploy import DPNN

# ind 0 is the width, ind 1 is the height
CAMERA_SIZE = (1280, 720)
CROP_BORDER_SIZE = (40, 10)
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

class main:
    def __init__(self):
        self.cam_hand = CameraHandler(CAMERA_SIZE, CROP_BORDER_SIZE, DOWNSCALE_FACTOR, CAM_PORTS)
        self.dp_pc = DPNN(CAM_DIST, DISP_CAL, PRUNED_PC_SIZE)