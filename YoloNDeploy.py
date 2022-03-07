import torch
import numpy as np
import cv2
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

class YoloNDeploy:
    def __init__(self):
        self.device = select_device()
        print(self.device)
        self.model = DetectMultiBackend("best.pt", device=self.device)


    def deploy(self, cv2img):
        init_size = cv2img.shape
        imgs = cv2.resize(cv2img, (416, 416), interpolation = cv2.INTER_AREA)
        imgs = torch.tensor(imgs)
        imgs = imgs.cuda()
        imgs = imgs.type(torch.cuda.FloatTensor)
        imgs = torch.reshape(imgs, [1, 3, 416, 416])
        pred = self.model(imgs, augment = False, visualize = False)
        pred = non_max_suppression(pred, 0.25, 0.45, max_det = 1000)
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords((416,416), det[:, :4], (416,416)).round()
                print(det.shape)
                print(det[200,:])
                for *xyxy, conf, cls in reversed(det):
                    print(xyxy, conf)