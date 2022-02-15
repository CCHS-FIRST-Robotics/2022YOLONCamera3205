import cv2
from PIL import Image
import numpy as np


class CameraHandler:
    def __init__(self, CAMERA_SIZE, CROP_BORDER_SIZE, DOWNSCALE_FACTOR, CAM_PORTS):
        self.CAMERA_SIZE = CAMERA_SIZE
        self.CROP_BORDER_SIZE = CROP_BORDER_SIZE
        self.DOWNSCALE_FACTOR = DOWNSCALE_FACTOR
        self.initCams(CAM_PORTS)

    def initCams(self, CAM_PORTS):
        self.lcam = cv2.VideoCapture(CAM_PORTS[0])
        self.rcam = cv2.VideoCapture(CAM_PORTS[1])
        self.cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_READ)
        self.Left_Stereo_Map1 = self.cv_file.getNode("Left_Stereo_Map_x").mat()
        self.Left_Stereo_Map2 = self.cv_file.getNode("Left_Stereo_Map_y").mat()
        self.Right_Stereo_Map1 = self.cv_file.getNode("Right_Stereo_Map_x").mat()
        self.Right_Stereo_Map2 = self.cv_file.getNode("Right_Stereo_Map_y").mat()

    def processCam(self, snp):
        p_snp = Image.fromarray(np.uint8(snp)).convert('RGB')
        w, h = p_snp.size
        p_snp = p_snp.crop((self.CROP_BORDER_SIZE[0], self.CROP_BORDER_SIZE[1], w - self.CROP_BORDER_SIZE[0],
                            h - self.CROP_BORDER_SIZE[1]))
        w2, h2 = p_snp.size
        p_snp.resize((round(w2 * self.DOWNSCALE_FACTOR[0]), round(h2 * self.DOWNSCALE_FACTOR[1])))
        return np.asarry(p_snp)

    def snapshot(self):
        _, l_frame = self.lcam.read()
        _, r_frame = self.rcam.read()

        l_frame = cv2.remap(l_frame, self.Left_Stereo_Map1, self.Left_Stereo_Map2, cv2.INTER_LANCZOS4,
                            cv2.BORDER_CONSTANT, 0)
        r_frame = cv2.remap(r_frame, self.Right_Stereo_Map1, self.Right_Stereo_Map2, cv2.INTER_LANCZOS4,
                            cv2.BORDER_CONSTANT, 0)
        return self.processCam(l_frame), self.processCam(r_frame)