import cv2
from PIL import Image
import numpy as np


class CameraHandler:
    def __init__(self, CAMERA_SIZE, CROP_BORDER_SIZE, DOWNSCALE_FACTOR, CAM_PORTS, cal_set):
        self.CAMERA_SIZE = CAMERA_SIZE
        self.CROP_BORDER_SIZE = CROP_BORDER_SIZE
        self.DOWNSCALE_FACTOR = DOWNSCALE_FACTOR
        self.initCams(CAM_PORTS, cal_set)

    def initCams(self, CAM_PORTS, cal_set):
        self.lcam = cv2.VideoCapture(CAM_PORTS[0])
        self.rcam = cv2.VideoCapture(CAM_PORTS[1])
        self.cv_file = cv2.FileStorage(cal_set, cv2.FILE_STORAGE_READ)
        self.Left_Stereo_Map1 = self.cv_file.getNode("Left_Stereo_Map_x").mat()
        self.Left_Stereo_Map2 = self.cv_file.getNode("Left_Stereo_Map_y").mat()
        self.Right_Stereo_Map1 = self.cv_file.getNode("Right_Stereo_Map_x").mat()
        self.Right_Stereo_Map2 = self.cv_file.getNode("Right_Stereo_Map_y").mat()
        
        self.l_mat = np.float32(np.genfromtxt("old_mtxL.csv",delimiter = ','))#.astype(np.float32)#.mat()
        self.r_mat = np.float32(np.genfromtxt("old_mtxL.csv",delimiter = ','))#.astype(np.float32)#.mat()
        self.ln_mat = np.float32(np.genfromtxt("new_mtxL.csv", delimiter=','))#.astype(np.float32)#.mat()
        self.rn_mat = np.float32(np.genfromtxt("new_mtxR.csv", delimiter=','))#.astype(np.float32)#.mat()
        self.distL = np.float32(np.genfromtxt("distL.csv", delimiter=','))#.astype(np.float32)#.mat()
        self.distR = np.float32(np.genfromtxt("distR.csv", delimiter=','))#.astype(np.float32)
        
        #print(self.l_mat, self.distL, self.ln_mat)
        #self.mapxl, self.mapyl = cv2.initUndistortRectifyMap(self.l_mat,self.distL, None, self.ln_mat, (640, 480), 1, (640, 480))
        
        #self.mapxr, self.mapyr = cv2.initUndistortRectifyMap(self.r_mat,self.distR, None, self.rn_mat, (640, 480), 1, (640, 480))
        

    def processCam(self, snp):
        p_snp = Image.fromarray(np.uint8(snp)).convert('RGB')
        w, h = p_snp.size
        p_snp = p_snp.crop((self.CROP_BORDER_SIZE[0], self.CROP_BORDER_SIZE[1], w - self.CROP_BORDER_SIZE[0],
                            h - self.CROP_BORDER_SIZE[1]))
        w2, h2 = p_snp.size
        p_snp.resize((round(w2 * self.DOWNSCALE_FACTOR[0]), round(h2 * self.DOWNSCALE_FACTOR[1])))
        return np.asarray(p_snp)

    def snapshot(self):
        _, l_frame = self.lcam.read()
        _, r_frame = self.rcam.read()
        #l_frame = cv2.imread("image_l_6.jpg")
        #r_frame = cv2.imread("image_l_6.jpg")
        
        
        #l_frame = cv2.undistort(l_frame, self.l_mat, self.distL)
        #r_frame = cv2.undistort(r_frame, self.r_mat, self.distR)
        #l_frame = cv2.remap(l_frame, self.mapxl, self.mapyl, cv2.INTER_LINEAR)
        #r_frame = cv2.remap(r_frame, self.mapxr, self.mapyr, cv2.INTER_LINEAR)

        l_frame = cv2.remap(l_frame, self.Left_Stereo_Map1, self.Left_Stereo_Map2, cv2.INTER_NEAREST,
                            cv2.BORDER_CONSTANT, 0)
        r_frame = cv2.remap(r_frame, self.Right_Stereo_Map1, self.Right_Stereo_Map2, cv2.INTER_NEAREST,
                            cv2.BORDER_CONSTANT, 0)
        return l_frame, r_frame