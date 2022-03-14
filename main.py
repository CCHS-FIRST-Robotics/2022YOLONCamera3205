import numpy as np
import math
from CameraHandler import CameraHandler
from Odometry import Odometry
from display import Display
from BallTrack.BallTrack import BallTrack
import cv2
import time
import YoloNDeploy
from VisualPosEst import ball2coords
from CNetworkTable import CNetworkTable
from Ball2LPos import Ball2LPos
from MonocularOdometry import MonocularOdometry
from ImageSave import ImageSave

# ind 0 is the width, ind 1 is the height
CAMERA_SIZE = (1280, 720)
#CROP_BORDER_SIZE = (32, 48)
CROP_BORDER_SIZE = (0, 0)
DOWNSCALE_FACTOR = (1, 1)
# ind 0 is Left, ind 2 is Right
CAM_PORTS = (2, 1)

CAM_DIST = 0.214

X_FOV = (2 * math.pi) * 53 / 360

DISP_CAL = np.array([[1., 0., 0., -292.3226819],
                     [0., 1., 0., -228.78747177],
                     [0., 0., 0., 499.35464466],
                     [0., 0., -6.80272109, 0.]])

PRUNED_PC_SIZE = 2500

AREA_PROP = 0.7
COL_PROP = 0.2

R_COL = [0, 76, 51, 9, 255, 255]
B_COL = [95, 106, 65, 132, 255, 255]

BALL_RADIUS = 0.12

LOCAL_POS = [-0.127, 0.2794, 0.5]

OBS_H_RANGE = [BALL_RADIUS + 0.03, 0.6]
GRID_SIZE = 0.1
TOWER_RAD = 1
ROBOT_RAD = 0.4


class main:
    def __init__(self):
        self.cam_hand = CameraHandler(CAMERA_SIZE, CROP_BORDER_SIZE, DOWNSCALE_FACTOR, CAM_PORTS, "improved_params2.xml")
        #self.cam_hand = CameraHandler(CAMERA_SIZE, CROP_BORDER_SIZE, DOWNSCALE_FACTOR, CAM_PORTS, "home_test.xml")
        self.ball_track = BallTrack(R_COL, B_COL, AREA_PROP, COL_PROP, BALL_RADIUS, X_FOV, [0.3, 1.2])
        self.odo = Odometry(LOCAL_POS)
        self.display = Display()
        self.yolon = YoloNDeploy.YoloNDeploy()
        self.networkT = CNetworkTable()
        self.ball2pos = Ball2LPos((480, 640))
        cam_mat = np.genfromtxt("new_mtxL.csv", delimiter=',')
        cam_mat = np.eye(3)
        #self.monoOdo = MonocularOdometry(cam_mat = cam_mat)
        self.save = ImageSave()

    def update(self):
        start_time = time.time()
        l, r = self.cam_hand.snapshot()
        #l = cv2.imread("image_40.jpg")
        #r = cv2.imread("image_40.jpg")
        self.ball2pos.img_shape = l.shape
        lball, rball = self.yolon.deploy(l, r)
        nball_list = self.ball2pos.makeBallList(lball, rball)
        ball_list = self.ball_track.updateTrack(nball_list, l.shape, self.odo)
        self.display.display(l, r, lball, rball)
        #self.monoOdo.process_frame(l)
        #self.monoOdo.visual_odometery()
        if (len(lball) + len(rball) > 0):
            self.save.save(l, r)
        for ball in ball_list:
            if (ball.state != 0):
                print("pos", ball.pos)
        #print("Pos", self.monoOdo.get_mono_coordinates())
        #print(ball_list)
        print("Elapsed Time: {}".format(time.time() - start_time))
        self.networkT.updateNetwork(self.odo, ball_list)
        
        
m = main()
if __name__ == "__main__":
    while True:
        m.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()